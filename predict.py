import os
import sys
import torch
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper

def defineArgs():
    parser = ArgumentParser()
    parser.add_argument('--model_fn', type=str, default='best.ckpt')
    parser.add_argument('--pretrained_name', type=str, default='beomi/kcbert-base')
    return parser.parse_args()

def init(config):
    tokenizer = KCBertTokenizerWrapper(config.pretrained_name, 0, None).tokenizer
    model = KCBertClassifier.load_from_checkpoint(   
        checkpoint_path=os.path.join(os.getcwd(),'checkpoints/kcbert',config.model_fn),
        bert_name=config.pretrained_name, 
        n_classes=2, 
        max_epoch=0, 
        batch_size=0, 
        lr=0, 
        eps=0, 
        warmup_ratio=0, 
        bert_freeze=False)

    return tokenizer, model
    
def predict(text, tokenizer, model):
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
    tokenizer, model = init(config)
    
    for line in sys.stdin:
        predict(line, tokenizer, model)
    