import torchaudio as ta
import pytorch_lightning as pl
import torchaudio.transforms as T
import torch
import numpy as np
import yaml
# import mir_eval
import gc

import warnings
# from train import RTBWETrain
# from datamodule import *
from utils import draw_spec

from tqdm import tqdm
import wandb
from pesq import pesq
from pystoi import stoi
import random
from torch.utils.data import Subset
import soundfile as sf
from datetime import datetime
import sys
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
# from SEANet_v2 import SEANet_ver2

from dataset import FeatureExtractorDataset
# from model_decoder import resnet
# from model_encoder import 
from model import AutoEncoder
from matplotlib import pyplot as plt
import os
import argparse

DEVICE = f'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"DEVICE: {DEVICE}")

## Dictionary to store all models and information
TPARAMS = {}
NOTES = 'BESSL_Encoder'
# config = yaml.load(open("./configs/exp1.yaml", 'r'), Loader=yaml.FullLoader)
START_DATE = NOTES +'_' + datetime.now().strftime("%Y%m%d-%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser(description="AutoEncoder Training Script")
    parser.add_argument('--config', type=str, required=True, help="Path to the config file")
    args = parser.parse_args()
    return args

def wandb_log(loglist, epoch, note):
    for key, val in loglist.items():
        if isinstance(val, torch.Tensor):
            item = val.cpu().detach().numpy()
        else:
            item = val
        try:
            if isinstance(item, float):
                log = item
            elif isinstance(item, plt.Figure):
                log = wandb.Image(item)
                plt.close(item)
            elif item.ndim in [2, 3]:  # Image
                log = wandb.Image(item, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            elif item.ndim == 1:  # Audio
                log = wandb.Audio(item, sample_rate=16000, caption=f"{note.capitalize()} {key.capitalize()} Epoch {epoch}")
            else:
                log = item
        except Exception as e:
            print(f"Failed to log {key}: {e}")
            log = item

        wandb.log({
            f"{note.capitalize()} {key.capitalize()}": log,
        }, step=epoch)

#########################
from torch.cuda.amp import autocast

def train_step(train_parameters):
    train_parameters['model'].train()
    result = {}
    result['loss'] = 0
    ## MSE Loss

    train_bar = tqdm(train_parameters['train_dataloader'], desc="Train", position=1, leave=False, disable=False)
    criterion = torch.nn.MSELoss()

    # Train DataLoader Loop
    for spec_e, masked_spec_e,_ in train_bar:
        masked_spec_e = masked_spec_e.to(DEVICE)
        spec_e = spec_e.to(DEVICE)
        
        # Forward
        recon = train_parameters['model'](masked_spec_e)

        # Loss
        loss = criterion(recon, spec_e)

        train_parameters['optimizer'].zero_grad()
        loss.backward()
        train_parameters['optimizer'].step()

        # Loss
        result['loss'] += loss.item()

        # 
        train_bar.set_postfix({
                'MSE Loss': f'{loss.item():.4f}',
            })
        del masked_spec_e, spec_e, recon, loss
        torch.cuda.empty_cache()
        gc.collect()

    train_bar.close()
    result['loss'] /= len(train_parameters['train_dataloader'])

    return result

def test_step(test_parameters, store_lr_hr=False):
    test_parameters['model'].eval()  # 모델을 평가 모드로 전환
    result = {}
    result['loss'] = 0  # AutoEncoder의 MSE Loss를 저장할 변수

    test_bar = tqdm(test_parameters['val_dataloader'], desc='Validation', position=1, leave=False, disable=False)

    criterion = torch.nn.MSELoss()

    i = 0
    # Test DataLoader Loop
    with torch.no_grad():
        for spec_e, masked_spec_e,_ in test_bar:
            i += 1
            masked_spec_e = masked_spec_e.to(DEVICE)
            spec_e = spec_e.to(DEVICE)

            # Forward pass
            recon = test_parameters['model'](masked_spec_e)

            # Compute Loss (MSE)
            loss = criterion(recon, spec_e)
            result['loss'] += loss.item()

            test_bar.set_postfix({
                'MSE Loss': f'{loss.item():.4f}',
            })

            # Optional: Store spectrograms for specific indices
            if i in [5, 50, 500] and store_lr_hr:
                key_suffix = f'_{i}'
                result[f'audio_input{key_suffix}'] = draw_spec(masked_spec_e.squeeze().cpu().numpy(), seglen=2, sample_rate=48000, freq_range=[6,31])
                result[f'audio_target{key_suffix}'] = draw_spec(spec_e.squeeze().cpu().numpy(), seglen=2, sample_rate=48000, freq_range=[6,31])
                result[f'audio_recon{key_suffix}'] = draw_spec(recon.squeeze().cpu().numpy(), seglen=2, sample_rate=48000, freq_range=[6,31])

            # Clear Out
            del masked_spec_e, spec_e, recon, loss
            torch.cuda.empty_cache()
            gc.collect()

        test_bar.close()
        result['loss'] /= len(test_parameters['val_dataloader'])  # 평균 Loss 계산

    return result

def main():
    ################ Read Config Files
    torch.manual_seed(42)
    args = parse_args()
    config = yaml.load(open(args.config, 'r'), Loader=yaml.FullLoader)

    wandb.init(project='BESSL_p2_D4',
           entity='woongzip1',
           config=config,
           name=START_DATE,
           notes=NOTES)

    ################ Load Dataset
    dataset = FeatureExtractorDataset(path_dir_wb=config['dataset']['path_dir'], spectrogram_dir=config['dataset']['path_spec'], 
                                      seg_len=config['dataset']['seg_len'], mask_fraction=0.5)
    dataset_size = len(dataset)
    train_size = int(0.995 * dataset_size)
    test_size = dataset_size - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    # _, test_dataset = random_split(test_dataset, [0.999, 0.001])
    
    print(f'Train Dataset size: {len(train_dataset)} | Validation Dataset size: {len(test_dataset)}\n')

    TPARAMS['train_dataloader'] = DataLoader(train_dataset, batch_size = config['dataset']['batch_size'], 
                                            # sampler = train_sampler,
                                            num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True,
                                            pin_memory=True)
    TPARAMS['val_dataloader'] = DataLoader(test_dataset, batch_size=1, shuffle=False,
                                            num_workers=config['dataset']['num_workers'], prefetch_factor=2, persistent_workers=True,
                                            pin_memory=True)

    print(f"DataLoader Loaded!: {len(TPARAMS['train_dataloader'])} | {len(TPARAMS['val_dataloader'])}")

    ################ Load Models
    TPARAMS['model'] = AutoEncoder(in_channels=config['model']['in_channels'])    
    TPARAMS['model'].to(DEVICE)
    print("Model Loaded!")

    ################ Load Optimizers
    TPARAMS['optimizer'] = torch.optim.Adam(TPARAMS['model'].parameters(), lr=config['optim']['learning_rate'], 
                                            betas=(config['optim']['B1'],config['optim']['B2']))
    
    ################ Load Checkpoint if available
    start_epoch = 1

    if config['train']['ckpt']:
        checkpoint_path = config['train']['ckpt_path']
        if os.path.isfile(checkpoint_path):
            start_epoch = load_checkpoint(TPARAMS['model'], TPARAMS['optimizer'], checkpoint_path)
        else:
            print(f"Checkpoint file not found at {checkpoint_path}. Starting training from scratch.")
        
    torch.manual_seed(42)
    ################ Training Loop
    print('Train Start!')
    BAR = tqdm(range(start_epoch, config['train']['max_epochs'] + 1), position=0, leave=True)
    best_MSE = 1e10

    store_lr_hr = True  # Flag to store spectrograms initially
    for epoch in BAR:
        TPARAMS['current_epoch'] = epoch
        train_result = train_step(TPARAMS)
        wandb_log(train_result, epoch, 'train')

        if epoch % config['train']['val_epoch'] == 0:
            # Validation step
            val_result = test_step(TPARAMS, store_lr_hr)
            wandb_log(val_result, epoch, 'val')

            if store_lr_hr:
                store_lr_hr = False  # Store only for the first validation

            # Save best model based on MSE Loss
            if val_result['loss'] < best_MSE:
                best_MSE = val_result['loss']
                save_checkpoint(TPARAMS['model'], epoch, best_MSE, config)

            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Train Loss: {train_result['loss']:.4f}, "
                    f"Val Loss: {val_result['loss']:.4f}")

        else:
            desc = (f"Epoch [{epoch}/{config['train']['max_epochs']}] "
                    f"Train Loss: {train_result['loss']:.4f}")

        BAR.set_description(desc)

    gc.collect()
    final_epoch = config['train']['max_epochs']
    save_checkpoint(TPARAMS['model'], final_epoch, val_result['loss'], config)


def save_checkpoint(model, epoch, loss, config):
    checkpoint_dir = config['train']['ckpt_save_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch}_mse_{loss:.3f}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': TPARAMS['optimizer'].state_dict(),  # Save optimizer state
        'loss': loss,
    }, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def load_checkpoint(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Move optimizer state to GPU if necessary
    for state in optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(DEVICE)
    print("CHECKPOINT LOADED! **** ")
    return checkpoint['epoch']
if __name__ == "__main__":
    main()
