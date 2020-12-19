import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from src import DataModule, TIMM, MyEarlyStopping

size = 384
config = {
    # optimization
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 64,
    # data
    'extra_data': 1,
    'subset': 0,
    'num_workers': 20,
    # model
    'backbone': 'vit_base_patch32_384',
    # data augmentation
    'size': size,
    'train_trans': {
        'PadIfNeeded': {
            'min_height': size,
            'min_width': size
        },
        'RandomCrop': {
            'height': size, 
            'width': size
        },
        'HorizontalFlip': {},
        'VerticalFlip': {},
        'Normalize': {}
    },
    'val_trans': {
        'PadIfNeeded': {
            'min_height': size,
            'min_width': size
        },
        'CenterCrop': {
            'height': size, 
            'width': size
        },
        'Normalize': {}
    },
    # training params
    'precision': 16,
    'max_epochs': 50,
    'val_batches': 1.0,
    'es_start_from': 0,
    'patience': 3
}

dm = DataModule(
    file = 'data_extra' if config['extra_data'] else 'data_old', 
    **config
)

model = TIMM(config)

#wandb_logger = WandbLogger(project="cassava", config=config)

es = MyEarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-{config["size"]}-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')

trainer = pl.Trainer(
    gpus=1,
    precision=config['precision'],
    #logger= wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)
