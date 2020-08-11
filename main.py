import torch
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
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
    parser.add_argument('--log_dir', type=str, default='/logs')
    parser.add_argument('--chk_dir', type=str, default='/checkpoints')

    parser = KCBertClassifier.add_model_specific_args(parser)
    
    return parser.parse_args()
    
def main(config):
    #model
    model = KCBertClassifier(config.bert_name, 
                             config.n_classes, 
                             config.max_epoch, 
                             config.batch_size, 
                             config.lr, 
                             config.eps, 
                             config.warmup_ratio,
                             config.bert_freeze)
    
    #dataloader
    trainset, validset = KCBertDataloader.loadData(config.train_fn, 
                                                   config.sep, 
                                                   config.text_idx, 
                                                   config.label_idx)
    
    label_encoder = LabelEncoder().fit(trainset.labels)
    wrapper = KCBertTokenizerWrapper(config.bert_name, 
                                     config.max_length, 
                                     label_encoder)
    
    train_loader = KCBertDataloader(trainset, 
                                    config.batch_size,
                                    wrapper.tokenizerWrapper)
    
    valid_loader = KCBertDataloader(validset, 
                                    config.batch_size, 
                                    wrapper.tokenizerWrapper) 

    #checkpoint
    checkpoint_callback = ModelCheckpoint(filepath=config.chk_dir,
                                          monitor='val_loss',
                                          save_top_k=1,
                                          mode='min')
    
    #logger
    logger = TensorBoardLogger(config.log_dir)
    
    #trainer
    trainer = None
    if torch.cuda.is_available():
        trainer = Trainer(gpus=config.gpus, checkpoint_callback=checkpoint_callback, logger=logger)  
    else:
        trainer = Trainer(checkpoint_callback=checkpoint_callback, logger=logger)

    trainer.fit(model, train_loader, valid_loader)
   
if __name__ == '__main__':
    config = defineArgs()
    main(config)