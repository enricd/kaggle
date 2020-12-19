from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class MyEarlyStopping(EarlyStopping):

    def on_validation_end(self, trainer, pl_module):
        if trainer.current_epoch < pl_module.hparams.start_from:
            print("Skipping early stopping until epoch ", pl_module.hparams.start_from)
            pass
        else:
            super().on_validation_end(trainer, pl_module)
