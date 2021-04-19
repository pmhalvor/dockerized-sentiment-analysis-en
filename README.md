# A sentiment analysis model deployed on Kubernetes cluster

This model provides a baseline architecture for a sentiment analysis task, using BERT to generate context embeddings of inputs. 

Users can server their own interactive endpoint by running `app.py`, then passing a JSON payload to : http://0.0.0.0/predict

The payload expects, a key-value pair "text":"sentence to be classified".

The return value of a properly formatted request is a prediction and certianty score.
- Predictions can be interpreted as 0=negative and 1=positive.
- Certianty score is out of 100.

---------------------
## Getting started with further development
This app is more or less ready to use right out of the box, with only a few steps required to get everything started.

### Step 1: Requirements
In the environment this repository is cloned to, call `pip install -r requirements.txt`. If PyTorch is not installed, this library will also need to be pip-installed. _(Note: Make sure your disk has enough space for the entire PyTorch package)_ 

### Step 2: Train locally
One of the two files missing from this repository is `models/SA_bert_state_dict.pt` due to GitHub's size limits (file size ca. 500MB). However, this file is easily created by calling `train.py`. The comments in this script guide users through the basic steps of training a neural netowrk using PyTorch. If training takes took long (or crashes due to memory errors), try reducing number of epochs. 

The data used to train the current deployment can be downloaded from: https://storage.googleapis.com/pmhalvor-public/data/stanford_sentiment_binary.tsv.gz

### Step 3: Serve `app.y`
The model file should be saved, meaning `app.py` can be run from your local machine. Then in your favorite API client, build a request with a JSON body containing:
```
{
"text":"The sentence you'd like classified"
}
```
Send it to `0.0.0.0/predict` and enjoy!

---------------------

## Files
- `app.py`: Flask file currently serving the app
- `model.py`: Customizable BERT model
- `preprocess.py`: Helper function for loading dataset
- `train.py`: Script for simple training, further fine tuning




(Currently still in development)
