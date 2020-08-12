import os
import sys
import torch
from argparse import ArgumentParser
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertTokenizerWrapper
from flask import Flask, request, jsonify


app = Flask(__name__)

@app.route('/api/v1/start', methods=['GET', 'POST'])
def start():    
    if request.method == 'POST':
        return {
            'code':'SUCCESS',
            'payload':'post'
        }
    else:
        return {
            'code':'SUCCESS',
            'payload':{
                'model_list':['beomi/kcbert-base']
            }
        }
 
if __name__ == "__main__":
    app.run()
