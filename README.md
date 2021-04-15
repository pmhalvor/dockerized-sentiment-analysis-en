# A sentiment analysis model deployed on Kubernetes cluster

This model provides a baseline architecture for a sentiment analysis task, using BERT to generate context embeddings of inputs. 

Users can interact with the model via the API endpoint: http://35.202.26.96/predict

A body payload is needed, with a key-value "text":"setence to be classified", where 0=negative and 1=positive.

(Currently still in development)
