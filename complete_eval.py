import os
from Data_Preparation.data_preparation import Data_Preparation
import numpy as np
from scipy.signal import hilbert, chirp, stft
import yaml
from diffusion import DDPM
from unet import UNet
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm, trange
from utils import train, evaluate, metrics
from torch.utils.data import DataLoader, TensorDataset
import torch_dct as dct
import torch.nn.functional as F
from tabulate import tabulate
import matplotlib.pyplot as plt

if not os.path.exists('./images'):
    os.makedirs('./images')

if __name__ == "__main__": 
    shot_sg = [1,3,5,10]
    device = 'cuda:0'
    
    for shots in shot_sg:
        
        ssd_total = []
        mad_total = []
        prd_total = []
        cos_sim_total = []
        snr_noise = []
        snr_recon = []
        snr_improvement = []
        n_level = []
        
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
        output_path = "./model.pth"
        
        model.load_state_dict(torch.load(output_path, weights_only=False))
        
        # due to the stochasticity of the synthesized dataset, 
        # we suggest saving the dataset in advance for better reproducibility
        x_test_path = "./X_test.pt"
        y_test_path = "./y_test.pt"
        [_, _, X_test, y_test] = Data_Preparation(1)

        if os.path.exists(x_test_path) and os.path.exists(y_test_path):
            X_test = torch.load(x_test_path)
            y_test = torch.load(y_test_path)
            print("Loaded X_test and y_test from local files")
        else:
            [_, _, X_test, y_test] = Data_Preparation(1)
            torch.save(X_test, x_test_path)
            torch.save(y_test, y_test_path)
            print("Generated and saved X_test and y_test to local files")
        

        X_test = X_test.permute(0,2,1) 
        y_test = y_test.permute(0,2,1)
        X_test = X_test.detach()
        test_set = TensorDataset(y_test, X_test)
        
        test_loader = DataLoader(test_set, batch_size=50, num_workers=0)
        
        n_level.append(np.load('rnd_test.npy'))
        
        with tqdm(test_loader) as it:
            for batch_no, (clean_batch, noisy_batch) in enumerate(it, start=1):
                clean_batch, noisy_batch = clean_batch.to(device), noisy_batch.to(device)
                noisy_batch_0 = noisy_batch.clone()
                noisy_batch = dct.dct(noisy_batch, norm='ortho')[:,:, :1000] / 3
            
                if shots > 1:
                    output = 0
                    for i in range(shots):
                        output+=model.denoising(noisy_batch)
                    output /= shots
                else:
                    output = model.denoising(noisy_batch)
                
                noisy_batch = noisy_batch_0
                output = F.pad(output, (0, 2600), mode='constant', value=0) * 3
                output = dct.idct(output, norm='ortho')
                
                clean_batch = clean_batch.permute(0, 2, 1)
                noisy_batch = noisy_batch.permute(0, 2, 1)
                output = output.permute(0, 2, 1)
                
                out_numpy = output.cpu().detach().numpy()
                clean_numpy = clean_batch.cpu().detach().numpy()
                noisy_numpy = noisy_batch.cpu().detach().numpy() 

                # Visualization            
                plt.figure(figsize=(10, 4))
                plt.plot(clean_numpy[1, :, 0], label='Clean ECG', color='blue')
                plt.plot(noisy_numpy[1, :, 0], label='Noisy ECG', color='red')
                plt.plot(out_numpy[1, :, 0], label='Denoised ECG', color='green')
                plt.title(f'ECG Signals - Batch {batch_no}, Sample 1')
                plt.xlabel('Time')
                plt.ylabel('Amplitude')
                plt.legend()
                plt.grid(True)
                save_path = f'./images/batch_{batch_no}_sample_1.png'
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(save_path)
                plt.close()                

                ssd_total.append(metrics.SSD(clean_numpy, out_numpy))
                mad_total.append(metrics.MAD(clean_numpy, out_numpy))
                prd_total.append(metrics.PRD(clean_numpy, out_numpy))
                cos_sim_total.append(metrics.COS_SIM(clean_numpy, out_numpy))
                snr_noise.append(metrics.SNR(clean_numpy, noisy_numpy))
                snr_recon.append(metrics.SNR(clean_numpy, out_numpy))
                snr_improvement.append(metrics.SNR_improvement(noisy_numpy, out_numpy, clean_numpy))
                    
                    
        ssd_total = np.concatenate(ssd_total, axis=0)
        mad_total = np.concatenate(mad_total, axis=0)
        prd_total = np.concatenate(prd_total, axis=0)
        cos_sim_total = np.concatenate(cos_sim_total, axis=0)
        snr_noise = np.concatenate(snr_noise, axis=0)
        snr_recon = np.concatenate(snr_recon, axis=0)
        snr_improvement = np.concatenate(snr_improvement, axis=0)
        n_level = np.concatenate(n_level, axis=0)
        
        
        segs = [0.2, 0.6, 1.0, 1.5, 2.0]

        table_data = [
            ["ssd", f"{ssd_total.mean():.3f} ± {ssd_total.std():.3f}"],
            ["mad", f"{mad_total.mean():.3f} ± {mad_total.std():.3f}"],
            ["prd", f"{prd_total.mean():.3f} ± {prd_total.std():.3f}"],
            ["cos_sim", f"{cos_sim_total.mean():.3f} ± {cos_sim_total.std():.3f}"],
            ["snr_in", f"{snr_noise.mean():.3f} ± {snr_noise.std():.3f}"],
            ["snr_out", f"{snr_recon.mean():.3f} ± {snr_recon.std():.3f}"],
            ["snr_improve", f"{snr_improvement.mean():.3f} ± {snr_improvement.std():.3f}"]
        ]

        print(f"******************{shots}-shots******************")
        print("******************ALL******************")
        print(tabulate(table_data, headers=["Metric", "Mean ± Std"], tablefmt="pretty"))
        
        metric_dict = {
            "ssd": ssd_total,
            "mad": mad_total,
            "prd": prd_total,
            "cos_sim": cos_sim_total,
            "snr_in": snr_noise,
            "snr_out": snr_recon,
            "snr_improve": snr_improvement
        }

        headers = ["Metric"] + [f"{segs[idx_seg]} < noise < {segs[idx_seg + 1]}" for idx_seg in range(len(segs) - 1)]

        table_data = []
        for metric, data in metric_dict.items():
            row = [metric]
            for idx_seg in range(len(segs) - 1):
                idx = np.argwhere(np.logical_and(n_level >= segs[idx_seg], n_level <= segs[idx_seg + 1]))

                metric_data = data[idx]
                row.append(f"{metric_data.mean():.3f} ± {metric_data.std():.3f}")

            table_data.append(row)

        print("******************Noise Segments Metrics******************")
        print(tabulate(table_data, headers=headers, tablefmt="pretty"))