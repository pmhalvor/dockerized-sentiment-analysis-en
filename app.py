import json
import torch
import requests
import os, io

from flask import Flask
from flask import request
from flask_restful import Api
from flask_restful import Resource
from model import Bert
from google.cloud import storage

# local paths to config files
params_path = '/models/SA_bert_params.pt'
state_dict_path = '/models/SA_bert_state_dict.pt'
score_path = '/models/SA_bert_scores.pt'

# cloud paths to config files 
# params_path = os.environ['bert_params']
# state_dict_path = os.environ['bert_state_dict']
# score_path = os.environ['bert_score']
# CLOUD_STORAGE_BUCKET = os.environ['CLOUD_STORAGE_BUCKET']


class Predict(Resource):
    def __init__(self):
        self.model = self.load_model()
        self.hello = "hello there"

    def load_model(self):
        params = path_to_loaded_torch(params_path)
        params['device'] = 'cpu' # force back to cpu (hope this is enough!)
        model = Bert(**params)
        model.load_state_dict(path_to_loaded_torch(state_dict_path))
        return model

    def post(self):
        payload = request.get_json()
        text = payload['text']

        prediction, certianty = self.model.predict(text)

        return (prediction, certianty*100), # certianty score in percentages

def path_to_loaded_torch(filepath):
    '''
    Handling local files in cloud 
        (best solution I could think of)
    '''
    storage_client = storage.Client()
    bucket = storage_client.bucket('gcp-terraform-pytorch-bucket-1')
    blob = bucket.blob(filepath)
    bcontent = blob.download_as_bytes()
    data = io.BytesIO(bcontent)
    return torch.load(data, map_location=torch.device('cpu'))


######### FLASK ##########
app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.run(host='0.0.0.0')