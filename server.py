import os
import sys
import torch
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper
from flask import Flask, request, jsonify, g
from predict import define_args, init, predict

class DotConfig(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as k:
            raise AttributeError(k)
    
    def __setattr__(self, key, value):
        self[key] = value
    
    def __delattr__(self, key):
        try:
            del self[key]
        except KeyError as k:
            raise AttributeError(k)
    
    def __repr__(self):
        return '<DotConfig '+dict.__repr__(self)+'>'

def load():
    config = DotConfig({'encoder_model':'beomi/kcbert-base',
                        'chk_dir':'checkpoints',
                        'chk_fn':'epoch=2-val_loss=0.38.ckpt',
                        'max_length':128})
    
    return init(config)

model, tokenizer, label_vocab = load()
app = Flask(__name__)

@app.route('/api/v1/classification', methods=['POST'])
def classification():
    post_param = request.get_json()
    text = post_param['text']
    
    result = predict(text, label_vocab, tokenizer, model)
    
    return {
        'code':'SUCESS',
        'result':result
    }

if __name__ == "__main__":
    print('Run server on port 5000 ...')
    app.run()
