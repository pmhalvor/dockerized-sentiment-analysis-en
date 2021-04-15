# A sentiment analysis model deployed on Kubernetes cluster

This model provides a baseline architecture for a sentiment analysis task, using BERT to generate context embeddings of inputs. 

Users can interact with the model via POST requests to the API endpoint: http://35.202.26.96/predict

A JSON payload is expected, with a key-value "text":"setence to be classified".

The return value of a properly formatted request is a prediction and certianty score.
- Predictions can be interpreted as 0=negative and 1=positive.
- Certianty score is out of 100.

---------------------
## Files
- `app.py`: Flask file currently serving the app
- `model.py`: Customizable BERT model
- `preprocess.py`: Helper function for loading dataset
- `train.py`: Script for simple training, further fine tuning




(Currently still in development)
