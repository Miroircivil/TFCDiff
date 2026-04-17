import argparse
import torch
import datetime
import yaml
import os
import shutil
import numpy as np

from Data_Preparation.data_preparation import Data_Preparation

from diffusion import DDPM
from unet import UNet
from utils import train, evaluate

from torch.utils.data import DataLoader, Subset, TensorDataset

from sklearn.model_selection import train_test_split

# backup
def backup_py_files():
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_dir = os.path.join('./backup', current_time)
    os.makedirs(backup_dir, exist_ok=True)
    
    source_dirs = ['.', './Data_Preparation']
    for source_dir in source_dirs:
        if os.path.exists(source_dir):
            for file in os.listdir(source_dir):
                file_path = os.path.join(source_dir, file)
                if os.path.isfile(file_path) and file.endswith('.py'):
                    shutil.copy2(file_path, backup_dir)
    
    config_file = './config/base.yaml'
    if os.path.exists(config_file):
        shutil.copy2(config_file, backup_dir)
    else:
        print(f"error: {config_file}")
    
if __name__ == "__main__": 
    current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    backup_py_files()
    parser = argparse.ArgumentParser(description="TFCDiff for ECG denoising")
    parser.add_argument('--device', default='cuda:0', help='Device')
    parser.add_argument('--n_type', type=int, default=1, help='noise version')
    args = parser.parse_args()
    
    path = "./config/base.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    foldername = "./check_points/noise_type_" + str(args.n_type) + "_" + current_time + "/"
    print('folder:', foldername)
    os.makedirs(foldername, exist_ok=True)
    
    # obtain dataset
    data = Data_Preparation(args.n_type)
    for i, tensor in enumerate(data):
        data[i] = tensor.permute(0, 2, 1)
    X_train, y_train, X_test, y_test = data
    train_val_set = TensorDataset(y_train, X_train)
    test_set = TensorDataset(y_test, X_test)
    
    # split dataset
    train_idx, val_idx = train_test_split(list(range(len(train_val_set))), test_size=0.3)
    train_set = Subset(train_val_set, train_idx)
    val_set = Subset(train_val_set, val_idx)
    train_loader = DataLoader(train_set, batch_size=config['train']['batch_size'],
                              shuffle=True, drop_last=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=config['train']['batch_size'], drop_last=True, num_workers=0)
    test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
    
    # initialize model
    unet_config = config["unet_config"]
    base_model = UNet(
        in_channel=unet_config["in_channel"],
        out_channel=unet_config["out_channel"],
        inner_channel=unet_config["inner_channel"],
        channel_mults=unet_config["channel_mults"],
        attn_res=unet_config["attn_res"],
        res_blocks=unet_config["res_blocks"],
        dropout=unet_config["dropout"],
        seq_len=unet_config["seq_len"],
        norm_groups=unet_config["norm_groups"]
    ).to(args.device)
    model = DDPM(base_model, config, args.device)
    
    #train
    train(model, config['train'], train_loader, args.device, 
          valid_loader=val_loader, valid_epoch_interval=1, foldername=foldername)
    
    #eval final with 1-generation
    print('eval final')
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #eval best with 1-generation
    print('eval best')
    output_path = foldername + "/model.pth"
    model.load_state_dict(torch.load(output_path))
    evaluate(model, val_loader, 1, args.device, foldername=foldername)
    
    #eval test with 1-generation
    print('eval test')
    evaluate(model, test_loader, 1, args.device, foldername=foldername)
    
    
    
    
    
    
    
    
    
    
