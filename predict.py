import os
import sys
import torch
import torch.nn as nn
import re
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper
from transformers import AutoTokenizer
from transformers import BertForSequenceClassification

def define_args():
    parser = ArgumentParser()
    parser.add_argument('--task'         , type=str, default='aicc')
    parser.add_argument('--encoder_model', type=str, default='beomi/kcbert-base')
    parser.add_argument('--max_length'   , type=int, default=128)
    parser.add_argument('--chk_dir'      , type=str, default='checkpoints')
    parser.add_argument('--chk_fn'       , type=str, default='epoch=2-val_loss=0.38.ckpt')
    parser.add_argument('--topk'         , type=int, default=5)
    return parser.parse_args()

class SentimentClassification():
    def __init__(self, config):
        self.config = config

    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
            
    def initialize(self):
        tokenizer   = KCBertTokenizerWrapper(self.config, None).tokenizer
        checkpoint  = torch.load(os.path.join(self.config.chk_dir, self.config.chk_fn), map_location=self.get_device())
        
        state_dict  = checkpoint['state_dict']
        hparams     = checkpoint['hparams']
        label_vocab = checkpoint['label_vocab']

        model = KCBertClassifier(hparams)
        model.load_state_dict(state_dict)
        
        self.model = model
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab

    def predict(self, text):
        self.model.eval()
        
        with torch.no_grad():
            encoded_input = self.tokenizer(text, 
                                    padding=True,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')

            input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
            output = self.model(input_ids, attention_mask).flatten().tolist()
            label_vocab_list = self.label_vocab.tolist()

            output_list = sorted(list(zip(label_vocab_list, output)), key=lambda x: x[1], reverse=True) 
            
            result = []
            for idx, data in enumerate(output_list):
                result.append({
                    'order':idx,
                    'category':data[0],
                    'weight':data[1]
                })
            
        return result


class AICCClassification():
    def __init__(self, config):
        self.config = config
        self.name = 'aicc'
    def get_device(self):
        if torch.cuda.is_available():
            return torch.device('cuda:0')
        else:
            return torch.device('cpu')
            
    def initialize(self):
        checkpoint  = torch.load(os.path.join(self.config.chk_dir, self.config.chk_fn), map_location=self.get_device())
        tokenizer   = AutoTokenizer.from_pretrained(self.config.encoder_model)
        
        state_dict  = checkpoint['bert']
        label_vocab = checkpoint['classes']

        model = BertForSequenceClassification.from_pretrained(
            self.config.encoder_model,
            num_labels=len(label_vocab)
        )
        model.load_state_dict(state_dict)
        
        self.model = model
        self.tokenizer = tokenizer
        self.label_vocab = label_vocab

    def preprocess(self, text):
        lines = []
    
        for line in [text]:
            if line.strip() != '':
                temp = re.sub(r'[○▶\-\+=,#/\?:^$.@*\"※~&%ㆍ!』\\‘|\(\)\[\]\<\>`\'…》\r\n\t＊→★▲▼■□ㅇ_;]', ' ', line.strip())
                pattern = re.compile(r'\s\s+')
                temp = re.sub(pattern, ' ', temp)
                lines += [temp]
        
        return lines

    def predict(self, text):
        self.model.eval()
        
        with torch.no_grad():
            texts = self.preprocess(text)
            encoded_input = self.tokenizer(texts, 
                                    padding=True,
                                    truncation=True,
                                    return_attention_mask=True,
                                    return_tensors='pt')

            input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
            output = self.model(input_ids, attention_mask=attention_mask)[0]
            output = nn.Softmax(dim=1)(output)
            propbs, indice = output.cpu().topk(self.config.topk)
            
            result = []
            for idx, data in enumerate(zip(indice[0].tolist(), propbs[0].tolist())):
                result.append({
                    'order':idx,
                    'category':self.label_vocab[data[0]],
                    'weight':data[1]
                })
            
        return result

if __name__ == '__main__':
    config = define_args()
    task = None

    if config.task == 'sentiment':
        task = SentimentClassification(config)
        task.initialize()
    elif config.task == 'aicc':
        task = AICCClassification(config)
        task.initialize()
    else:
        print('Unknown task')
    
    if task != None:
        for line in sys.stdin:
            result = task.predict(line)
            print(result)