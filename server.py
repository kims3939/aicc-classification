import os
import json
from flask import Flask, request
from predict import SentimentClassification, AICCClassification

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

def load(model_name):
    if model_name == 'kcbert_classifier':
        config = DotConfig({'encoder_model':'beomi/kcbert-base',
                            'chk_dir':'checkpoints',
                            'chk_fn':'kcbert_80.pth',
                            'data_dir':'dataset',
                            'label_fn':'label.tsv',
                            'topk':5,
                            'max_length':128})
        task = AICCClassification(config)
        task.initialize()
        return task

task = load('kcbert_classifier')
app = Flask(__name__, static_folder='./frontend/build', static_url_path='/')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/start', methods=['GET','POST'])
def start():
    if request.method == 'GET':
        return {
            'code':'SUCCESS',
            'result':{
                'available_models':[
                    'kcbert_classifier'
                ]
            }
        }

    if request.method == 'POST':
        post_param = request.get_data().decode('utf-8')
        post_param = json.loads(post_param)
        model_name = post_param['model_name']
        name = load(model_name)

        return {
            'code':'SUCCESS',
            'result':{
                'task_name':name
            }
        }

@app.route('/api/v1/classification', methods=['POST'])
def classification():
    post_param = request.get_data().decode('utf-8')
    post_param = json.loads(post_param)
    text = post_param['text']
    result = task.predict(text)
    
    return {
        'code':'SUCESS',
        'result':result
    }

if __name__ == "__main__":
    print('Run server on port 5000 ...')
    app.run(host="0.0.0.0", port="5000")
