import torch
from torch import nn
from torch import optim
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.functional import accuracy
from transformers import AutoModel
from transformers import get_linear_schedule_with_warmup
from argparse import ArgumentParser

class KCBertClassifier(LightningModule):
    
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        
        #model hyper-params
        parser.add_argument('--encoder_model'  , type=str, default='beomi/kcbert-base')
        parser.add_argument('--n_classes'      , type=int, required=True)
        parser.add_argument('--freeze_encoder' , action='store_true')
        parser.add_argument('--max_length'     , type=int, default=128)
    
        #optimizer hyper-params
        parser.add_argument('--lr'           , type=float, default=5e-5)
        parser.add_argument('--eps'          , type=float, default=1e-8)
        parser.add_argument('--warmup_ratio' , type=float, default=.1)

        return parser

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.label_vocab = None
        self.__build_model()

    def __build_model(self):
        self.encoder   = AutoModel.from_pretrained(self.hparams.encoder_model)
        self.generator = nn.Linear(self.encoder.config.hidden_size, self.hparams.n_classes)
        self.softmax   = nn.Softmax(dim=-1)

        if self.hparams.freeze_encoder:
            self.freeze(self.encoder)
        
    def freeze(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        x = self.encoder(input_ids, attention_mask)[0]
        x = self.generator(x[:,0,:])
        x = self.softmax(x)

        return x
      
    def training_step(self, batch, batch_index):
        input_ids, attention_mask, target = batch
        
        y_hat = self(input_ids, attention_mask)
        train_loss = nn.functional.cross_entropy(y_hat, target)
        train_acc  = accuracy(y_hat, target)
        
        logs = {'train_loss':train_loss, 'train_acc':train_acc}

        return {'loss':train_loss, 'log':logs, 'progress_bar':logs}
    
    def validation_step(self, batch, batch_index):
        input_ids, attention_mask, target = batch

        y_hat = self(input_ids, attention_mask)
        return {'y_hat':y_hat, 'target':target}
    
    def validation_epoch_end(self, outputs):
        y_hat  = [output['y_hat']  for output in outputs]
        target = [output['target'] for output in outputs]

        y_hat  = torch.cat(y_hat,0)
        target = torch.cat(target, 0)
        
        val_loss = nn.functional.cross_entropy(y_hat, target)
        val_acc  = accuracy(y_hat, target)
        logs = {'val_loss':val_loss, 'val_acc':val_acc}

        return {'log':logs, 'progress_bar':logs}
    
    def configure_optimizers(self):
        total_step  = self.hparams.max_epochs * self.hparams.batch_size
        warmup_step = self.hparams.warmup_ratio * total_step

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {
                'params': [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                'weight_decay': 0.01
            },
            {
                'params': [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                'weight_decay': 0.0
            }
        ]

        optimizer = optim.AdamW(optimizer_grouped_parameters, lr=self.hparams.lr, eps=self.hparams.eps)
        shceduelr = get_linear_schedule_with_warmup(optimizer, warmup_step, total_step)
        return [optimizer], [shceduelr]
    
    def on_save_checkpoint(self, checkpoint):
        checkpoint['hparams'] = self.hparams
        checkpoint['label_vocab'] = self.label_vocab
