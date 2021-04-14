import json
import torch
import requests
import torch.nn.functional as F

from flask import Flask
from flask import request
from flask_restful import Api
from flask_restful import Resource
from torchvision import transforms

# local paths to config files
params_path = '/models/SA_bert_params.pt'
state_dict_path = '/models/SA_bert_state_dict.pt'
score_path = '/models/SA_bert_scores.pt'

class Predict(Resource):
    def __init__(self):
        self.model = self.load_model()

    def load_model(self):
        params = torch.load(params_path)
        model = Bert(**params)
        model.load_state_dict(torch.load(state_dict_path))
        return model

    def post(self):
        payload = request.get_json()
        text = payload['text']

        self.model.eval()
        prediction, certianty = self.model.predict(text)

        return (prediction, certianty*100), # certianty score in percentages



app = Flask(__name__)
api = Api(app)

api.add_resource(Predict, '/predict')

if __name__ == "__main__":
    app.run(host='0.0.0.0')