import os
import sys
import torch
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper

def define_args():
    parser = ArgumentParser()
    parser.add_argument('--encoder_model', type=str, default='beomi/kcbert-base')
    parser.add_argument('--max_length'   , type=int, deafult=128)
    parser.add_argument('--chk_dir'      , type=str, default='checkpoints/')
    parser.add_argument('--chk_fn'       , type=str, default='best_model.ckpt')
    
    return parser.parse_args()

def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    else:
        return torch.device('cpu')

def init(config):
    tokenizer   = KCBertTokenizerWrapper(config, None).tokenizer
    checkpoint  = torch.load(os.path.join(config.chk_dir, config.chk_fn), map_location=get_device())
    
    state_dict  = checkpoint['state_dict']
    hparams     = checkpoint['hparams']
    label_vocab = checkpoint['label_vocab']

    model = KCBertClassifier(hparams)
    model.load_state_dict(state_dict)

    return (model, tokenizer, label_vocab)
    
def predict(text, label_vocab, tokenizer, model):
    model.eval()
    encoded_input = tokenizer(text, 
                              padding=True,
                              truncation=True,
                              return_attention_mask=True,
                              return_tensors='pt')

    input_ids, attention_mask = encoded_input['input_ids'], encoded_input['attention_mask']
    output = model(input_ids, attention_mask)
    output = torch.argmax(output, dim=-1)

    print('RESULT: ',label_vocab[output])

if __name__ == '__main__':
    config = define_args()
    model, tokenizer, label_vocab = init(config)
    
    for line in sys.stdin:
        predict(line, label_vocab, tokenizer, model)