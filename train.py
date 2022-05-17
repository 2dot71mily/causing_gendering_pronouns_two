from transformers import TrainingArguments, Trainer
from transformers import AutoConfig
from datasets import load_dataset
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from pathlib import PurePath
from huggingface_hub import ModelHubMixin
from transformers.modeling_outputs import TokenClassifierOutput
from transformers import AutoModelForTokenClassification, AutoTokenizer, AutoTokenizer
import wandb

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Consts for users to adjust
TEST_RUN = True  # For end to end test on tiny dataset, till overfitting
PERCENT_DATASET = 100  # Reduce for normal training run with smaller dataset portion
USE_WIKIBIO = False  # False if using 'reddit' dataset.
MODELS_OUTPUT_DIR = 'models'
DATASETS_OUTPUT_DIR = 'datasets'


# Consts that result from above choices
DATASET = 'wiki_bio' if USE_WIKIBIO else 'reddit'
N_EPOCHS = 1 if not TEST_RUN else 2  # 00 #EMILY
METADATA_COLS = ['none', 'subreddit'] if DATASET is "reddit" else [
    'none', 'birth_date']
P_DATASET = PERCENT_DATASET if not TEST_RUN else "na"


MODEL_CHECKPOINT = 'bert-base-uncased'
EPS = 1e-5  # to avoid /0 errors

# Tokenizer and labeling consts
MULTITOKEN_WOMAN_WORD = 'policewoman'
MULTITOKEN_MAN_WORD = 'spiderman'
NON_LOSS_TOKEN_ID = -100
NON_GENDERED_TOKEN_ID = 30  # Picked an int that will pop out visually

# Picked an int that will pop out visually
LABEL_DICT = {'female': 9, 'male': -9}
CLASSES = list(LABEL_DICT.keys())

MAX_TOKEN_LENGTH = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_gendered_token_ids(tokenizer):

    # Set up gendered token constants
    gendered_lists = [
        ['he', 'she'],
        ['him', 'her'],
        ['his', 'hers'],
        ["himself", "herself"],
        ['male', 'female'],
        ['man', 'woman'],
        ['men', 'women'],
        ["husband", "wife"],
        ['father', 'mother'],
        ['boyfriend', 'girlfriend'],
        ['brother', 'sister'],
        ["actor", "actress"],
    ]

    # Generating dicts here for potential later token reconstruction of predictions
    male_gendered_dict = {list[0]: list for list in gendered_lists}
    female_gendered_dict = {list[1]: list for list in gendered_lists}

    male_gendered_token_ids = tokenizer.convert_tokens_to_ids(
        list(male_gendered_dict.keys()))
    female_gendered_token_ids = tokenizer.convert_tokens_to_ids(
        list(female_gendered_dict.keys())
    )

    # Below technique is used to grab second token in a multi-token word
    # There must be a better way...
    multiword_woman_token_ids = tokenizer.encode(
        MULTITOKEN_WOMAN_WORD, add_special_tokens=False)
    assert len(multiword_woman_token_ids) == 2
    subword_woman_token_id = multiword_woman_token_ids[1]

    multiword_man_token_ids = tokenizer.encode(
        MULTITOKEN_MAN_WORD, add_special_tokens=False)
    assert len(multiword_man_token_ids) == 2
    subword_man_token_id = multiword_man_token_ids[1]

    male_gendered_token_ids.append(subword_man_token_id)
    female_gendered_token_ids.append(subword_woman_token_id)

    # Confirming all tokens are in vocab
    assert tokenizer.unk_token_id not in male_gendered_token_ids
    assert tokenizer.unk_token_id not in female_gendered_token_ids

    return male_gendered_token_ids, female_gendered_token_ids


def tokenize_and_append_metadata(batch, tokenizer, text_col, conditioning_key, text_marker=None, meta_col=None):
    mask_token_id = tokenizer.mask_token_id

    label_list = list(LABEL_DICT.values())
    assert label_list[0] == LABEL_DICT['female'], "LABEL_DICT not an ordered dict"
    label2id = {label: idx for idx, label in enumerate(label_list)}

    texts = []

    if USE_WIKIBIO:
        for i, full_text in enumerate(batch[text_col]):
            # Removing the initial text data, which largely overlaps with the meta-data,
            # which we are attempting to control-for in this experiment
            # This may seem unfair, but in many training scenarios, long texts like those
            # in wikipedia are chunked into segments, separating such initial (meta-data rich)
            # text from later segments, in a similar manner.
            text = full_text.split(text_marker)[-1]

            try:
                sample_meta_data = batch[meta_col][i]['table']
                for key in METADATA_COLS:
                    if key == 'none':
                        continue
                    key_idx = sample_meta_data['column_header'].index(key)
                    if conditioning_key == key:
                        conditioning_key_idx = key_idx

            except ValueError:
                continue  # Removing rows missing meta_data

            assert conditioning_key == 'none' or conditioning_key_idx is not None

            if conditioning_key != 'none':
                meta_data = sample_meta_data['content'][conditioning_key_idx]
                meta_data = f"{' '.join(conditioning_key.split('_'))} {meta_data}"
            else:
                meta_data = ''

            texts.append(f'{meta_data} {text}')

    else:  # for DATASET == REDDIT_DATASET:
        for i, text in enumerate(batch[text_col]):
            if conditioning_key != 'none':
                text = f"reddit: {batch[conditioning_key][i]}, {text}"
            texts.append(text)

    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=MAX_TOKEN_LENGTH,
    )

    filtered_tokenized = {
        'input_ids': [],
        'attention_mask': [],
        'labels': [],
    }

    for i, token_ids in enumerate(tokenized['input_ids']):

        # Finding the gender pronouns in the tokens
        female_tags = torch.tensor(
            [LABEL_DICT['female'] if id in female_gendered_token_ids else NON_GENDERED_TOKEN_ID for id in token_ids])
        male_tags = torch.tensor(
            [LABEL_DICT['male'] if id in male_gendered_token_ids else NON_GENDERED_TOKEN_ID for id in token_ids])

        # Removing rows that have no gendered pronouns
        if LABEL_DICT['female'] not in female_tags and LABEL_DICT['male'] not in male_tags:
            continue

        # Labeling and masking out occurrences of gendered pronouns
        labels = torch.tensor([NON_LOSS_TOKEN_ID] * len(token_ids))
        labels = torch.where(
            female_tags == LABEL_DICT['female'], label2id[LABEL_DICT['female']], NON_LOSS_TOKEN_ID)
        labels = torch.where(
            male_tags == LABEL_DICT['male'], label2id[LABEL_DICT['male']], labels)
        masked_token_ids = torch.where(
            female_tags == LABEL_DICT['female'], mask_token_id, torch.tensor(token_ids))
        masked_token_ids = torch.where(
            male_tags == LABEL_DICT['male'], mask_token_id, masked_token_ids)

        filtered_tokenized['input_ids'].append(masked_token_ids)
        filtered_tokenized['attention_mask'].append(
            tokenized['attention_mask'][i])
        filtered_tokenized['labels'].append(labels)

    return filtered_tokenized


def compute_metrics(p):
    predictions, labels = p

    labels = labels.flatten()
    mask = labels != -100
    labels = labels[mask]

    predictions = np.argmax(predictions, axis=2)
    predictions = predictions.flatten()
    predictions = predictions[mask]

    metrics = {}
    precision = []
    recall = []
    for class_idx, class_name in enumerate(CLASSES):
        precision.append(get_class_precision(predictions, labels, class_idx))
        metrics[f"{class_name}_precision"] = precision[class_idx]
        recall.append(get_class_recall(predictions, labels, class_idx))
        metrics[f"{class_name}_recall"] = recall[class_idx]

    wandb.log({
        f"{CLASSES[0]}_precision":  metrics[f"{CLASSES[0]}_precision"],
        f"{CLASSES[1]}_precision":  metrics[f"{CLASSES[1]}_precision"],
        f"{CLASSES[0]}_recall":  metrics[f"{CLASSES[0]}_recall"],
        f"{CLASSES[1]}_recall":  metrics[f"{CLASSES[1]}_recall"],
        "conf_mat": wandb.plot.confusion_matrix(
            probs=None,
            preds=predictions,
            y_true=labels,
            class_names=CLASSES),
    })
    return metrics


def get_class_precision(predictions, labels, class_type):
    class_fp = [
        int(p == class_type and l != class_type)
        for (p, l) in zip(predictions, labels)
        if l != 100
    ]
    tp_sum = get_class_true_positives(predictions, labels, class_type)

    return tp_sum / (tp_sum + np.array(class_fp).sum())


def get_class_recall(predictions, labels, class_type):
    class_fn = [
        int(p != class_type and l == class_type)
        for (p, l) in zip(predictions, labels)
        if l != 100
    ]
    tp_sum = get_class_true_positives(predictions, labels, class_type)
    return tp_sum / (tp_sum + np.array(class_fn).sum())


def get_class_true_positives(predictions, labels, class_type):
    class_tp = [
        int(p == class_type and l == class_type)
        for (p, l) in zip(predictions, labels)
        if l != 100
    ]
    return np.array(class_tp).sum() + EPS


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_CHECKPOINT, add_prefix_space=True)
    # get gendered_token_ids
    male_gendered_token_ids, female_gendered_token_ids = get_gendered_token_ids(
        tokenizer)

    dataset = None
    # tokenized_dataset = None
    for var in METADATA_COLS:
        run_name = f'cond_ft_{var}_on_{DATASET}__prcnt_{P_DATASET}__test_run_{TEST_RUN}'
        wandb.init(project=f"cond_ft_{DATASET}", config={
            "run_name": run_name}, reinit=True)

        try: 
            tokenized_dataset = load_dataset(run_name)
        except:
            if dataset is None:
                if TEST_RUN:
                    dataset = load_dataset(DATASET, split='train[:100]')
                elif P_DATASET != 100:
                    dataset = load_dataset(DATASET, split=f'train[:{P_DATASET}%]')
                else:
                    dataset = load_dataset(DATASET)

            tok_kwargs = {
                "tokenizer": tokenizer,
                'conditioning_key': var,
                "text_col": 'target_text' if USE_WIKIBIO else 'normalizedBody',
                'text_marker': '-rrb-' if USE_WIKIBIO else None,
                'meta_col': 'input_text' if USE_WIKIBIO else None,
            }
            tokenized_dataset = dataset.map(
                tokenize_and_append_metadata, batched=True, fn_kwargs=tok_kwargs,
                remove_columns=dataset.column_names
            )
            # TODO: load this if exists! # EMILY
            tokenized_dataset.save_to_disk(
                str(PurePath(DATASETS_OUTPUT_DIR, run_name)))
            tokenized_dataset.push_to_hub(run_name)
                                

        # initialize base model and tokenizer
        base_model = AutoModelForTokenClassification.from_pretrained(
            MODEL_CHECKPOINT)

        val_set_name = 'val'
        if (TEST_RUN or P_DATASET != 100) or not USE_WIKIBIO:
            # Then we are not using dataset maintainer's pre-defined splits, or no split provided
            tokenized_dataset = tokenized_dataset.shuffle(
                seed=42).train_test_split(test_size=0.05)
            val_set_name = 'test'

        args = TrainingArguments(
            str(PurePath(MODELS_OUTPUT_DIR, run_name)),
            evaluation_strategy="epoch",
            learning_rate=1e-5,
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            num_train_epochs=N_EPOCHS,
            weight_decay=1e-5,
            save_total_limit=3,
            logging_steps=10,
            lr_scheduler_type='linear',
            report_to="wandb",
            run_name=run_name,
            push_to_hub=True
        )

        trainer = Trainer(
            model=base_model,
            args=args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset[val_set_name],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        base_model.push_to_hub(run_name)
        tokenizer.push_to_hub(run_name)
        tokenized_dataset = None  # ready for next iteration
