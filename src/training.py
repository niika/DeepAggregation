import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
import torchvision
import optuna
import pickle


def training(train_loader, test_loader, val_loader, model, model_params, trainer_params):
    net = model(**model_params)
    trainer = pl.Trainer(**trainer_params)
    trainer.fit(net, train_loader, val_dataloaders=val_loader)
    test_loss = trainer.test(net, test_loader)[0]["test_loss"]

    torch.cuda.empty_cache()
    #torch.cuda.synchronize(device = torch.device('cuda:0'))
    return test_loss, net, trainer


def objective(trial: optuna.trial.Trial):
    torch.cuda.empty_cache()
    train_loader = None
    test_loader = None
    net = None
    trainer = None
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    #train_loaders = [DataLoader(train_dataset, batch_size=2, shuffle=True) for train_dataset in train_datasets] 
    test_loader = DataLoader(test_dataset,batch_size=10)
    planes  = trial.suggest_int("planes", low=8, high=24, step=8)
    downsampling_factor= trial.suggest_int("downsampling_factor", low=2, high=2, step=1)
    encoder_num_blocks = trial.suggest_int("encoder_num_blocks", low=5, high=20, step=5)
    decoder_num_blocks = trial.suggest_int("decoder_num_blocks", low=5, high=20, step=5)
    smooth_num_blocks  = trial.suggest_int("smooth_num_blocks", low=5, high=20, step=5)

    agg_block = trial.suggest_categorical("agg_block",["Attention", "Mean"])
    experiment_name = "specshadocc-" + agg_block
    
    if agg_block=="Attention":
        mode = trial.suggest_categorical("mode",["softmax","sum","mean"])
        agg_params = { 
            "mode": mode,
            "num_heads": trial.suggest_int("num_heads", low=2, high=8, step=2),
            "dim_feedforward": trial.suggest_int("dim_feedforward", low=64, high=128, step=64),
            "num_layers": trial.suggest_int("num_layers", low=2, high=4, step=1)
        }
        experiment_name = experiment_name +"-"+ mode 
    else:
        agg_params = {
            "mode": None,
            "num_heads": None,
            "dim_feedforward": None,
            "num_layers": None
        }
    
    #lr_monitor = LearningRateMonitor(logging_interval='step')
    
    oom = False
    try:
        net = DeepAggNet(planes=planes, downsampling_factor=downsampling_factor,
                         encoder_num_blocks=encoder_num_blocks, 
                         decoder_num_blocks=decoder_num_blocks,
                         smooth_num_blocks=smooth_num_blocks,
                         agg_block=agg_block, **agg_params)

        # Create Logger
        comet_logger = CometLogger(
            #api_key="tMEjeyq5M7v1IPRCvS5fyGyuo",
            workspace="semjon", # Optional
            project_name= COMET_ML_PROJECT, # Optional
            # rest_api_key=os.environ["COMET_REST_KEY"], 
            save_dir='./comet-ml',
            experiment_name=experiment_name, # Optional,
            display_summary_level = 0
        )

        early_stop = pl.callbacks.EarlyStopping('val_loss', patience=50)
        trainer = pl.Trainer(gpus=1,logger=comet_logger, max_epochs=300, 
                             progress_bar_refresh_rate=0 , #fast_dev_run=True,
                             callbacks=[early_stop])
        trainer.fit(net, train_loader, val_dataloaders=test_loader)
        test_loss = trainer.test(net, test_loader)[0]["test_loss"]

        # Free Memory
        #del trainer, test_loader, train_loader, net
        torch.cuda.empty_cache()
        #torch.cuda.synchronize(device = torch.device('cuda:0'))
        return test_loss, net, trainer
    
    except (RuntimeError):
        oom = True
        
    if oom:
        #torch.cuda.synchronize(device = torch.device('cuda:0'))
        del trainer, test_loader, train_loader, net
        torch.cuda.empty_cache()
        return 0.1

    


# sampler = optuna.samplers.TPESampler(seed=10)

# study = optuna.create_study(direction='minimize', sampler=sampler)
# study.optimize(objective, n_trials=50,catch=(RuntimeError,), gc_after_trial=True)
# pickle.dump( study, open( "study.p", "wb" ) )