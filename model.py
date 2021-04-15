from itertools import chain
from torch.utils.data import Dataset
from transformers import BertModel, BertTokenizer

import numpy as np 
import torch


class Bert(torch.nn.Module):
    def __init__(
        self, lr=0.01, momentum=0.9, device='cpu', epochs=10,
        fine_tune_bert=False,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.epochs = epochs

        # BERT model and tokenizer (context embedder)
        self._bert = BertModel.from_pretrained('bert-base-uncased').to(
            self.device
        )
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        
        # customizable head w/ fixed in/out size (in: 768, out:1)
        self._head = torch.nn.Sequential(
            # add any torch.nn layer here (see: https://pytorch.org/docs/stable/nn.html)
            torch.nn.Linear(768, 1), 
            torch.nn.Sigmoid() # want inputs between 0 and 1
        ).to(self.device)

        # to (dis)activate externally 
        if fine_tune_bert:
            opt_params = chain(
                self._bert.parameters(),
                self._head.parameters(),
            ) # parameters optimizer will update in backward step
        else:
            opt_params = self._head.parameters()


        self._opt = torch.optim.AdamW(
            opt_params,
            lr=lr,     
        ) # can try SGD, Adam, etc. (see: https://pytorch.org/docs/stable/optim.html)
        
        self._loss = torch.nn.BCELoss() # will need to change if shapes change

    def tokenize(self, X):
        '''
        Convert sentence to embedding ids.

        X: list(str) containing sentences as strings
        
        return: tensor w/ shape (len(X), longest tokenized sentence)
        '''
        return self._tokenizer(
            X, padding=True, return_tensors='pt')['input_ids'].to(self.device)

    def forward(self, X):
        '''
        Forward step of model.

        X: tensor w/ shape (batch size, longest tokenized sentence)

        return: tensor of model output w/ shape (batch size)
        '''
        b = self._bert(X)
        pooler = b.pooler_output # gather whole sequence to single estimate
        return self._head(pooler).to(self.device)

    def backward(self, output, target):
        '''
        Optimization step of model. 

        output: tensor returned from forward w/ shape (batch size)
        target: tensor of labels for batch w/ shape (batch size)

        return: tensor of single loss score for batch shape (1)
        '''
        loss = self._loss(output, target.float())
        self._opt.zero_grad()
        loss.backward()
        self._opt.step()
        return loss

    def fit(self, loader, epochs=None, val_data=None):
        '''
        Train model using input dataloader.

        loader: DataLoader object (see train.py:32-35)
        epochs: int specifying how many epochs to run
        val_data: tuple(text, label) for evaulation each epoch
            default = None
        '''
        if not epochs:
            epochs = self.epochs
        self.train()
        print('training...')
        for epoch in range(epochs):
            epoch_loss, N = 0, 0
            for n, batch in enumerate(loader):
                tokens = self.tokenize(batch[0]) # first element of batch is list of strings
                output = self.forward(tokens.to(self.device)) 
                loss = self.backward(output, batch[1].to(self.device))
                epoch_loss += loss 
                N += 1
                print(f'batch: {n:3}\tloss: {loss:.7f}') if n%30==0 else None
            print(f'Epoch {epoch}\tavg loss: {epoch_loss/N:.7f}')
            if val_data:
                text, label = val_data
                self.validate(text, label, False)
        
        if self.device == torch.device('cuda'):
            torch.cuda.empty_cache()

    def validate(self, text, labels, verbose=True):
        '''
        Validate predictions on strings against true labels

        text: list(str)
        label: list(int) {0, 1}

        return: float with accuracy score
        '''
        print('validating...') if verbose else None
        with torch.no_grad():
            tokens = self.tokenize(text)
            output = self.forward(tokens).squeeze(1)
            
            compare = output.round() == labels.to(self.device)

            score = sum(compare).item()/len(output)
            print(f'\t% correct in validation: {score:.7f}')  
            self.train()
        return score

    def predict(self, text):
        '''
        Make prediction on list of inputs

        text: list(str) make sure input is list, not single string

        prediction: most likely predicted label
        confidence: how certian the model is on prediction
        '''
        self.eval()
        print('validating...')

        tokens = self.tokenize(text)
        output = self.forward(tokens)

        prediction = output.round().item()
        confidence = output.item() if prediction==1 else (1-output.item())
        return prediction, confidence


