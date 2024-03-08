import os
import hydra
import omegaconf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from copy import deepcopy
import pandas as pd
import numpy as np

import wandb
from src.utils.lightning import init_lightning_callbacks, set_seed

torch.set_float32_matmul_precision("medium")


def run(cfg: DictConfig):
    pl.seed_everything(cfg.seed, workers=True)
    set_seed(cfg.seed)
    print("Working directory : {}".format(os.getcwd()))
    config = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
    logger = instantiate(cfg.logger)(config=config)

    datamodule = instantiate(cfg.datamodule)
    dataset = datamodule.train_dataloader().dataset
    if dataset.num_features == 0:
        cfg.model.model.num_features = 1
    else:
        cfg.model.model.num_features = dataset.num_features
    cfg.model.model.num_classes = dataset.num_classes
    model = instantiate(cfg.model.model)

    checkpoint_callback = ModelCheckpoint(monitor="valid/acc", mode="max", save_top_k=1)

    trainer: pl.Trainer = instantiate(cfg.trainer, logger=logger, callbacks=[checkpoint_callback])
    trainer.validate(deepcopy(model), datamodule.val_dataloader())
    trainer.fit(model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)
    wandb.finish()
    return model.best_epoch, model.train_acc.cpu().numpy(), model.val_acc.cpu().numpy(), model.best_val_acc.cpu().numpy(), model.test_acc.cpu().numpy(), model.train_loss, model.step_size


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def run_model(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    run(cfg)


@hydra.main(config_path="conf", config_name="config", version_base="1.2")
def batch_run(cfg: DictConfig):

    # only run training for single configuration as defined in config file
    if cfg.batch_run == False:
        print(OmegaConf.to_yaml(cfg))
        run(cfg)

    # run training for all datasets with 10-fold cross-validation
    else:
        for dataset, deg_features in zip(['MUTAG', 'NCI1', 'PROTEINS', 'PTC_FM',
                        'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K',
                        'IMDB-BINARY', 'IMDB-MULTI', 'COLLAB', 'REDDIT-BINARY', 'REDDIT-MULTI-5K'],
                        [0, 0, 0, 0,
                        1, 1, 1, 1, 1,
                        0, 0, 0, 0, 0]):

            fold_inds = []
            epochs = []
            train_accs = []
            val_accs = []
            best_val_accs = []
            test_accs = []
            
            loss_per_step = []
            loss_per_epoch = []
            

            for fold_idx in range(10):

                cfg.dataset_name = dataset
                cfg.deg_features = deg_features
                cfg.fold_idx = fold_idx

                print(OmegaConf.to_yaml(cfg))
                best_epoch, train_acc, val_acc, best_val_acc, test_acc, train_loss, step_size = run(cfg)


                fold_inds.append(fold_idx)
                epochs.append(best_epoch)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                best_val_accs.append(best_val_acc)
                test_accs.append(test_acc)

                if cfg.save_learning_curves == True:
                    num_epochs = cfg.trainer.max_epochs
                    epoch_loss = []
                    k = int(len(train_loss)/num_epochs)
                    num_samples = np.sum(step_size)/num_epochs
                    for i in range(num_epochs):
                        loss_value = 0
                        for j in range(i,i+k):
                            loss_value += train_loss[j]*step_size[j]
                        loss_value /= num_samples
                        epoch_loss.append(loss_value)

                    loss_per_step.append(train_loss)
                    loss_per_epoch.append(epoch_loss)

                    loss_per_step_array = np.array(loss_per_step)
                    loss_per_epoch_array = np.array(loss_per_epoch)

                    if not os.path.exists("results/loss_per_step"): 
                        os.makedirs("results/loss_per_step") 
                    np.save(f'results/loss_per_step/{cfg.model.name}_{cfg.agg}_{dataset}_{cfg.deg_features}.npy', loss_per_step_array)

                    if not os.path.exists("results/loss_per_epoch"): 
                        os.makedirs("results/loss_per_epoch") 
                    np.save(f'results/loss_per_epoch/{cfg.model.name}_{cfg.agg}_{dataset}_{cfg.deg_features}.npy', loss_per_epoch_array)

            if cfg.save_results == True:

                if not os.path.exists("results"): 
                    os.makedirs("results") 

                results = pd.DataFrame({'fold_idx': fold_inds, 'test_epoch': epochs, 'train_acc': train_accs, 'val_acc': val_accs, 'best_val_acc': best_val_accs, 'test_acc': test_accs})
                results.to_csv(f'results/{cfg.model.name}_{cfg.agg}_{dataset}_{cfg.deg_features}.csv', index=False)


if __name__ == "__main__":
    batch_run()
