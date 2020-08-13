import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader, SequentialSampler, BatchSampler
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class KCBertDataloader(DataLoader):
    def __init__(self, dataset, batch_size, collate_fn):
        seq_sampler   = SequentialSampler(dataset)
        batch_sampler = BatchSampler(seq_sampler, batch_size, False)
        
        super().__init__(
            dataset=dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn
        )
    
    @staticmethod
    def loadData(hparams):
        train_dataset = pd.read_csv(os.path.join(hparams.train_dir, hparams.train_fn), sep=hparams.sep)
        valid_dataset = pd.read_csv(os.path.join(hparams.train_dir, hparams.valid_fn), sep=hparams.sep)
        
        train_x = train_dataset.iloc[:, hparams.text_idx].values.tolist()
        train_y = train_dataset.iloc[:, hparams.label_idx].values.tolist()

        valid_x = valid_dataset.iloc[:, hparams.text_idx].values.tolist()
        valid_y = valid_dataset.iloc[:, hparams.label_idx].values.tolist()

        trainset = KCBertDataset(train_x, train_y)
        validset = KCBertDataset(valid_x, valid_y)
        
        return trainset, validset
        

class KCBertDataset(Dataset):
    def __init__(self, texts, labels):
        super().__init__()
        self.texts  = []
        self.labels = []
        
        dataset = sorted(zip(texts, labels), key=lambda x: len(x[0]))
        for data in dataset:
            self.texts.append(data[0])
            self.labels.append(data[1])
        
    def __getitem__(self, index):
        text  = str(self.texts[index])
        label = str(self.labels[index])

        return {
            'text':text,
            'label':label
        }
    
    def __len__(self):
        return len(self.labels)

class KCBertTokenizerWrapper():
    def __init__(self, hparams, labelEncoder):
        self.tokenizer    = AutoTokenizer.from_pretrained(hparams.encoder_model)
        self.labelEncoder = labelEncoder
        self.max_length   = hparams.max_length

    def tokenizerWrapper(self, samples):
        texts  = [sample['text']  for sample in samples]
        labels = [sample['label'] for sample in samples]

        encoded_input = self.tokenizer(texts, 
                                       padding=True,
                                       truncation=True,
                                       max_length=self.max_length,
                                       return_attention_mask=True,
                                       return_tensors='pt')
        
        encoded_label = self.labelEncoder.transform(labels)
        
        return (encoded_input['input_ids'], 
                encoded_input['attention_mask'],
                torch.tensor(encoded_label, dtype=torch.long))