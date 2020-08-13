import torch
import os
from argparse import ArgumentParser
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from models.kcbert import KCBertClassifier
from dataloaders.kcbert import KCBertDataloader, KCBertTokenizerWrapper
from sklearn.preprocessing import LabelEncoder

def define_args():
    parser = ArgumentParser()
    
    #train param    
    parser.add_argument('--max_epochs' , type=int, default=3)
    parser.add_argument('--batch_size' , type=int, default=64)
    
    #device param
    parser.add_argument('--gpus', nargs='+', type=int)

    #data params
    parser.add_argument('--train_dir'  , type=str,   default='dataset/')
    parser.add_argument('--train_fn'   , type=str,   required=True)
    parser.add_argument('--valid_fn'   , type=str,   required=True)
    parser.add_argument('--label_idx'  , type=int,   required=True)
    parser.add_argument('--text_idx'   , type=int,   required=True)
    parser.add_argument('--sep'        , type=str,   default='\t')
    parser.add_argument('--valid_ratio', type=float, default=.3)
    

    #log params
    parser.add_argument('--prj_dir', type=str, required=True)
    parser.add_argument('--log_dir', type=str, default='logs/')
    parser.add_argument('--chk_dir', type=str, default='checkpoints/')
    
    #model params
    parser = KCBertClassifier.add_model_specific_args(parser)
    
    return parser.parse_args()
    
def main(hparams):
    #model
    model = KCBertClassifier(hparams)
    
    #dataset
    trainset, validset = KCBertDataloader.loadData(hparams)
    
    #label encoder
    label_encoder = LabelEncoder().fit(trainset.labels)
    model.label_vocab = label_encoder.classes_
    
    #dataloader
    wrapper = KCBertTokenizerWrapper(hparams, label_encoder)
    train_loader = KCBertDataloader(trainset, hparams.batch_size, wrapper.tokenizerWrapper)
    valid_loader = KCBertDataloader(validset, hparams.batch_size, wrapper.tokenizerWrapper) 

    #checkpoint
    checkpoint_callback = ModelCheckpoint(monitor='val_loss',
                                          save_top_k=1,
                                          mode='min',
                                          filepath=os.path.join(hparams.chk_dir, hparams.prj_dir, '{epoch}-{val_loss:.2f}'))
    
    #logger
    tb_logger = TensorBoardLogger(save_dir=hparams.log_dir, name=hparams.prj_dir)
    

    #trainer
    trainer = Trainer(gpus=hparams.gpus if torch.cuda.is_available() else None,
                      max_epochs=hparams.max_epochs,
                      logger=tb_logger,
                      checkpoint_callback=checkpoint_callback)
    
    trainer.fit(model, train_loader, valid_loader)
    
if __name__ == '__main__':
    hparams = define_args()
    main(hparams)