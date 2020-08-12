import os
import sys
import torch
import numpy as np
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper

def defineArgs():
    parser = ArgumentParser()
    parser.add_argument('--pretrained', type=str, default='beomi/kcbert-base')
    parser.add_argument('--chk_fn', type=str, default='best.ckpt')
    parser.add_argument('--label_path', type=str, default='label_dict')
    return parser.parse_args()

def init(config):
    tokenizer = KCBertTokenizerWrapper(config.pretrained, 0, None).tokenizer
    label_dict = np.load(os.path.join(os.getcwd(),config.label_path))
    
    model = KCBertClassifier.load_from_checkpoint(
        checkpoint_path=os.path.join(os.getcwd(), 'checkpoints', config.chk_fn),
        bert_name='beomi/kcbert-base',
        n_classes=2,
        max_epoch=1,
        batch_size=1,
        lr=0.0,
        eps=0.0,
        warmup_ratio=0.0,
        bert_freeze=False
    )

    return {
        'label_dict':label_dict,
        'tokenizer': tokenizer,
        'model': model
    }
    
def predict(text, label_dict, tokenizer, model):
    encoded_input = tokenizer(text, 
                              padding=True,
                              truncation=True,
                              return_attention_mask=True,
                              return_tensors='pt')

    input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
    output = model(input_ids, attention_mask)
    print(output)

if __name__ == '__main__':
    config = defineArgs()
    
    obj = init(config)
    model = obj['model']
    label_dict = obj['label_dict']
    tokenizer  = obj['tokenizer']

    for line in sys.stdin:
        predict(line, label_dict, tokenizer, model)