from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

import pandas as pd
import torch



class BertSentimentDataset(Dataset):

    def __init__(self, datapath, train_size=0.75, random_state=1, device="cpu") -> None:
        '''
        Loads tsv-data from provided path

        datapath: str specifying .tsv file localtion
            assumes tsv (csv w/ sep="\t") w/ header: tokens, label
        train_size: float specify split train/valid data, default 0.75
        random_state: int for reproduciblity
        device: str specifying where to cast y in __getitem__
        '''
        self.datapath = datapath
        self.device = device
        self.random_state = random_state

        print(f'processing data file {datapath}')
        df_train, df_test = self.process_raw_data(
            data_url=self.datapath,
            random_state=self.random_state,
            train_size=train_size,
            verbose=False,
        )

        self.text = list(df_train['tokens']) # list of strings
        self.label = list(df_train['label']) # list of bools

        self.text_test = list(df_test['tokens'])    # list of strings
        self.label_test = list(df_test['label'])    # list of bools

        self.training = True

    def __getitem__(self, index: int):
        '''
        Called when sliced (dataset[0]), spec. when DataLoader iterates

        index: int specifying desired element

        return: tuple(current_text:str, y:LongTensor) for single element at index
        '''
        current_text  = self.text[index]
        current_label = self.label[index]

        y = torch.LongTensor([current_label])

        if self.device == "cuda":
            y = y.to(torch.device("cuda"))

        return current_text, y

    def __len__(self):
        """
        Magic method to return the number of samples.
        """
        return len(self.text)

    @staticmethod
    def process_raw_data(
        data_url, 
        random_state=1,
        train_size=0.75, 
        verbose=True, 
    ):
        '''
        Read/process raw tab-seperated-value data

        Returns:
            pandas.DataFrame(labels:bool, tokens:str, lemmatized:str)
        '''

        df = pd.read_csv(data_url, sep='\t')
        print(f'data read with columns {df.columns}') \
            if verbose is True else None

        print('cleaning data...') if verbose is True else None
        # dropping wrong row
        df = df.drop(
            index=df[
                (df.label != "negative") & (df.label != "positive")
            ].index
        )

        df.label = df.label == 'positive'

        # checking balance of the classes
        print("Balance of the Classes:\n", df.label.value_counts(), "\n") \
            if verbose is True else None

        # fixing DF's indexes
        df.reset_index()

        print('train test splitting') if verbose is True else None
        # ########## splitting train and unseen test data
        df_train, df_test = train_test_split(
            df,
            train_size=train_size,
            random_state=random_state
        )
        del df


        return df_train, df_test
