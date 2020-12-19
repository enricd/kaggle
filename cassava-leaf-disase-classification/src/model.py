import pytorch_lightning as pl
import torchvision
import torch.nn.functional as F
from pytorch_lightning.metrics.functional.classification import accuracy
import torch
from torchvision import transforms
import timm
from pl_bolts.models.self_supervised import SwAV

class Model(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        x, y = batch
        if self.trainer.current_epoch < self.hparams.unfreeze:
            with torch.no_grad():
                features = self.backbone(x)
        else: 
            features = self.backbone(x)
        y_hat = self.head(features[-1])
        loss = F.cross_entropy(y_hat, y)
        acc = accuracy(y_hat, y)
        self.log('loss', loss)
        self.log('acc', acc, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = F.cross_entropy(y_hat, y)
        val_acc = accuracy(y_hat, y)
        self.log('val_loss', val_loss, prog_bar=True)
        self.log('val_acc', val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if self.hparams.scheduler:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer

class Resnet(Model):
    def __init__(self, config):
        super().__init__(config)
        self.resnet = getattr(torchvision.models, self.hparams.backbone)(pretrained=True)
        self.resnet.fc = torch.nn.Linear(self.resnet.fc.in_features, 5)
    
    def forward(self, x):
        return self.resnet(x)

class TIMM(Model):
    def __init__(self, config):
        super().__init__(config)
        self.backbone = timm.create_model(self.hparams.backbone, pretrained=self.hparams.pretrained, features_only=True)
        self.head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d(output_size=(1,1)),
            torch.nn.Flatten(),
            torch.nn.Linear(self.hparams.num_features, 5)
        )
    def forward(self, x):
        features = self.backbone(x)
        return self.head(features[-1])

class SWAV(Model):
    def __init__(self, config):
        super().__init__(config)
        weight_path = 'https://pl-bolts-weights.s3.us-east-2.amazonaws.com/swav/swav_imagenet/swav_imagenet.pth.tar'
        swav = SwAV.load_from_checkpoint(weight_path, strict=True)
        self.encoder = swav.model
        self.fc = torch.nn.Linear(3000, 5)

    def forward(self, x):
        features = self.encoder(x)[-1]
        return self.fc(features)
