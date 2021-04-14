from preprocess import BertSentimentDataset
from model import Bert 

import torch


############## CONFIG ##############
# configure training # TODO refactor to a config file for flexibility
load_saved_model  = False   # if False, new model trained from scratch
activate_training = True    # if True, will train for __ more epochs

epochs = 4
fine_tune_bert = True
force_save = False # NOTE Will overwrite current best model stored on disk
lr = 0.00001
momentum = 0.3

# local paths to config files
params_path     = '/models/SA_bert_params.pt'
state_dict_path = '/models/SA_bert_state_dict.pt'
score_path      = '/models/SA_bert_scores.pt'

# train on gpu or cpu
device = 'cuda' if torch.cuda.is_available() else 'cpu'
####################################


#############  TRAIN  ##############
# load data
dataset = BertSentimentDataset('stanford_sentiment_binary.tsv.gz',\
    train_size=0.90)
train_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=32,
)
valid_text  = list(dataset.text_test)
valid_label = torch.tensor(dataset.label_test)

# load/init model
if load_saved_model:
    print('Attemping model load...')
    params = torch.load(params_path)
    model = Bert(**params)
    model.load_state_dict(torch.load(state_dict_path))
    print('success!')
else:
    params = {
        'lr': lr,
        'momentum': momentum,
        'device':device,
        'fine_tune_bert':fine_tune_bert,
    }

    model = Bert(**params)


# train
if activate_training:
    print(f'Training on {device}...')
    model.fit(train_loader, epochs=epochs,\
        val_data=(valid_text, valid_label))
####################################


############  EVALUATE  ############
# prints valid score if verbose (default: True)
score = model.validate(valid_text, valid_label, verbose=True)

# autosaver based on previous score
saved_score = torch.load(score_path) # assumes score stored (crash if file does not exist)
if score>saved_score:
    save = True
else:
    save = False

if save or force_save:
    print('Saving...')
    torch.save(model.state_dict(), state_dict_path)
    torch.save(params, params_path)
    torch.save(score, score_path)
####################################


print('Complete!')
