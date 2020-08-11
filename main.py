import torch
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertDataloader, KCBertTokenizerWrapper
from sklearn.preprocessing import LabelEncoder

def defineArgs():
    parser = ArgumentParser()
    parser.add_argument('--gpus', nargs='+', type=int)
    parser.add_argument('--train_fn', type=str, required=True)
    parser.add_argument('--sep', type=str, default='\t')
    parser.add_argument('--label_idx', type=int, required=True)
    parser.add_argument('--text_idx', type=int, required=True)
    parser.add_argument('--valid_ratio', type=float, default=.3)
    parser = KCBertClassifier.add_model_specific_args(parser)
    return parser.parse_args()
    
def main(config):
    #model
    model = KCBertClassifier(config.bert_name, config.n_classes, config.max_epoch, config.batch_size, config.lr, config.eps, config.warmup_ratio)
    
    #dataloader
    trainset, validset = KCBertDataloader.loadData(config.train_fn, config.sep, config.text_idx, config.label_idx)
    label_encoder = LabelEncoder().fit(trainset.labels)
    wrapper = KCBertTokenizerWrapper(config.bert_name, config.max_length, label_encoder)
    train_loader = KCBertDataloader(trainset, config.batch_size, wrapper.tokenizerWrapper)
    valid_loader = KCBertDataloader(validset, config.batch_size, wrapper.tokenizerWrapper) 

    #train
    trainer = None
    if torch.cuda.is_available():
        trainer = Trainer(gpus=config.gpus)  
    else:
        trainer = Trainer()

    trainer.fit(model, train_loader, valid_loader)

if __name__ == '__main__':
    config = defineArgs()
    main(config)