from Data_Preparation.data_preparation import Data_Preparation
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import hilbert, chirp, stft
import yaml
from diffusion import DDPM
from unet import UNet
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from utils import train, evaluate, metrics
from torch.utils.data import DataLoader, Subset, ConcatDataset, TensorDataset
from scipy import stats
import os
import torch_dct as dct
import torch.nn.functional as F


if __name__ == "__main__": 
    shots = 1
    device = 'cuda:0'
    n_type = 1

    path = "config/base.yaml"
    with open(path, "r") as f:
        config = yaml.safe_load(f)

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
    ).to(device)
    model = DDPM(base_model, config, 'cuda:0')
    # we offer the pretrained weights
    output_path = "./check_points/model.pth"
    model.load_state_dict(torch.load(output_path, weights_only=False))

    # ensure its shape is (samples, 1, seq_len), and the amplitude is bounded by [-1, 1]
    X_test = np.load("yourdataset.npy")
    X_test = torch.FloatTensor(X_test)
    test_set = TensorDataset(X_test)

    test_loader = DataLoader(test_set, batch_size=10, num_workers=0, shuffle=False)
    eta = 3

    with tqdm(test_loader) as it:
        arrays_list = []
        for batch_no, noisy_batch in enumerate(it, start=1):
            noisy_batch = noisy_batch.to(device)
            noisy_batch = dct.dct(noisy_batch, norm='ortho')[:,:, :1000] / eta
            
            output = 0
            for i in range(shots):
                output+=model.denoising(noisy_batch)
                output /= shots
            
            output = F.pad(output, (0, 2600), mode='constant', value=0) * eta
            output = dct.idct(output, norm='ortho')
            output = output.permute(0, 2, 1)
            out_numpy = output.cpu().detach().numpy()
            arrays_list.append(out_numpy)
        concatenated_array = np.concatenate(arrays_list, axis=0)
        reshaped_array = concatenated_array.squeeze(axis=-1)
        # # if you split recordings into multiple samples, you may reshape here to recombine them
        # final_array = reshaped_array.reshape(recordings, -1)
        final_array = reshaped_array
        print("shape of the result", final_array.shape)
        save_directory = "./lpx/"
        save_path = os.path.join(save_directory, "denoised_signals.npy")
        np.save(save_path, final_array)



