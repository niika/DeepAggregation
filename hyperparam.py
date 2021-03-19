from comet_ml import Experiment
from pytorch_lightning.loggers import CometLogger
import datasets
import torch
from architectures import *
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import LearningRateMonitor
import optuna
import logging
import sys

optuna.logging.enable_default_handler()
logging.getLogger().setLevel(logging.INFO)  # Setup the root logger.
logging.getLogger().addHandler(logging.FileHandler('attention.log'))
optuna.logging.enable_propagation() 

# Configure Dataset
dataset = datasets.SpecShadOcc("SpecShadOcc/", 9)
test_length = 20
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [len(dataset)-test_length,test_length])
train_datasets = [datasets.SpecShadOccN(dataset, i) for i in range(1,9)]
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
#train_loaders = [DataLoader(train_dataset, batch_size=2, shuffle=True) for train_dataset in train_datasets] 
test_loader = DataLoader(test_dataset,batch_size=10)

def objective(trial: optuna.trial.Trial):

    planes  = trial.suggest_int("planes", low=8, high=24, step=8)
    downsampling_factor= trial.suggest_int("downsampling_factor", low=1, high=3, step=1)
    encoder_num_blocks = trial.suggest_int("encoder_num_blocks", low=5, high=10, step=2)
    decoder_num_blocks = trial.suggest_int("decoder_num_blocks", low=5, high=10, step=2)
    smooth_num_blocks  = trial.suggest_int("smooth_num_blocks", low=5, high=10, step=2)

    agg_block = trial.suggest_categorical("agg_block",["Attention"])
    
    if agg_block=="Attention":
        mode = trial.suggest_categorical("mode",["softmax","sum","mean"])
        agg_params = { 
            "mode": mode,
            "num_heads": trial.suggest_int("num_heads", low=4, high=8, step=4),
            "dim_feedforward": trial.suggest_int("dim_feedforward", low=64, high=64, step=64),
            "num_layers": trial.suggest_int("num_layers", low=2, high=4, step=1),
        }
        experiment_name = "specshadocc-" + "Att-" + mode
    else:
        agg_params = {}
        experiment_name = "specshadocc-" + "Mean"

    
    try:
        #lr_monitor = LearningRateMonitor(logging_interval='step')
        net = DeepAggNet(planes=planes, downsampling_factor=downsampling_factor,
                         encoder_num_blocks=encoder_num_blocks, 
                         decoder_num_blocks=decoder_num_blocks, smooth_num_blocks=smooth_num_blocks,
                         agg_block=agg_block, **agg_params)

        # Create Logger
        comet_logger = CometLogger(
            api_key="tMEjeyq5M7v1IPRCvS5fyGyuo",
            workspace="semjon", # Optional
            project_name="specshadocc-attention", # Optional
            # rest_api_key=os.environ["COMET_REST_KEY"], 
            save_dir='./hyperparameter',
            experiment_name="specshadocc-" + agg_block, # Optional,
            display_summary_level = 0
        )

        early_stop = pl.callbacks.EarlyStopping('val_loss', patience=5)
        trainer = pl.Trainer(gpus=1,logger=comet_logger, max_epochs=100, 
                             #fast_dev_run=True,
                             callbacks=[early_stop], 
                             progress_bar_refresh_rate=0)
        trainer.fit(net, train_loader, val_dataloaders=test_loader)
        return trainer.test(net, test_loader)[0]["test_loss"]
    
    except:
        print("Unexpected error:", sys.exc_info()[0])
        return 1.0
    

sampler = optuna.samplers.TPESampler(seed=10)

study = optuna.create_study(direction='minimize', sampler=sampler)
study.optimize(objective, n_trials=300)