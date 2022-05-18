# Training 

## Set up
```
git clone https://github.com/2dot71mily/causing_gendering_pronouns_two.git
python3 -m venv venv_cgp2
source venv_cgp2/bin/activate
apt-get install git-lfs // for linux or see below for other platforms
pip install --upgrade pip
pip install -r requirements.txt
huggingface-cli login  # paste in your hf write token at prompt
```

### Installing git-lfs on other platforms
```
// 1) Installing git-lfs on colab… 
// or is it still needed?  https://twitter.com/julien_c/status/1517231471739744256?s=20&t=T79DxcZmehwXWcNlCEDnHA
!curl -s https://packagecloud.io/install/repositories/github/git-lfs/script.deb.sh | sudo bash
!git lfs install
// from https://github.com/git-lfs/git-lfs/issues/3605#issuecomment-482591804
!sudo apt-get install git-lfs


// 2) Installing git-lfs on macOS
brew install git-lfs  // or port install git-lfs
```

## First run
Set the constants at top of file to:
- `TEST_RUN = True`, to confirm everything is working end to end and that you can train to overfitting.
- Select `USE_WIKIBIO = True` to train using [wiki-bio](https://huggingface.co/datasets/wiki_bio) or `False` to use  use [reddit](https://huggingface.co/datasetsreddit) dataset.

Then run:
```
python train.py
```
You model and dataset will be pushed to Hugging Face hub

Once things look ok, set `TEST_RUN = False`, and play with the other constants and the top of the file.


# Check out the results

This this WIP Hugging Face Space, [causing_gender_pronouns_two](https://huggingface.co/spaces/emilylearning/causing_gender_pronouns_two) to experiment with the outcomes of the models trained here.

Or check out this older and less expressive Hugging Face Space [causing_gender_pronouns](https://huggingface.co/spaces/emilylearning/causing_gender_pronouns) for more documentation details.








