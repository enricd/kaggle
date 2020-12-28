import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from src import Model, DataModule, MyEarlyStopping

size = 256
config = {
    # optimization
    'lr': 3e-4,
    'optimizer': 'Adam',
    'batch_size': 128,
    'scheduler': {
        'StepLR': {
            'step_size': 3,
            'gamma': 0.1,
            'verbose': True
        }
    },
    # data
    'extra_data': 1,
    'subset': 0.1,
    'num_workers': 20,
    # model
    'backbone': 'seresnext50_32x4d',
    'pretrained': True,
    'unfreeze': 0,
    # data augmentation
    'size': size,
    'train_trans': {
        'RandomCrop': {
            'height': size, 
            'width': size
        },
        'HorizontalFlip': {},
        'VerticalFlip': {},
        'Normalize': {}
    },
    'val_trans': {
        'CenterCrop': {
            'height': size, 
            'width': size
        },
        'Normalize': {}
    },
    # training params
    'precision': 16,
    'max_epochs': 50,
    'val_batches': 5,
    'es_start_from': 0,
    'patience': 3
}

dm = DataModule(
    file = 'data_extra' if config['extra_data'] else 'data_old', 
    **config
)

model = Model(config)

wandb_logger = WandbLogger(project="cassava-tl", name="step", config=config)

es = MyEarlyStopping(monitor='val_acc', mode='max', patience=config['patience'])
checkpoint = ModelCheckpoint(dirpath='./', filename=f'{config["backbone"]}-{config["size"]}-{{val_acc:.5f}}', save_top_k=1, monitor='val_acc', mode='max')
lr_monitor = LearningRateMonitor(logging_interval='step')

trainer = pl.Trainer(
    gpus=1,
    precision=config['precision'],
    logger= wandb_logger,
    max_epochs=config['max_epochs'],
    callbacks=[es, checkpoint, lr_monitor],
    limit_val_batches=config['val_batches']
)

trainer.fit(model, dm)
