import os
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper

def defineArgs():
    parser = ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--tokenizer_name', type=str, required=True)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--text', type=str, required=True)
    return parser.parse_args()

def predict(config):
    tokenizer = KCBertTokenizerWrapper(config.tokenizer_name, config.max_length).tokenizer
    encoded_input = tokenizer(config.text, 
                              padding=True,
                              truncation=True,
                              max_length=config.max_length,
                              return_attention_mask=True,
                              return_tensors='pt')
    
    input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
    model = KCBertClassifier.load_from_checkpoint(config.checkpoint_path)
    output = model(input_ids, attention_mask)
    print(output)

if __name__ == '__main__':
    config = defineArgs()
    predict(config)