# TFCDiff
## Updates 4/24/2024
1. Correction of Dataset Splitting Errors
A bug in the list within `data_preparation.py` previously caused two ECG records that should have been assigned to the test set to be incorrectly placed in the training set. This issue has now been fixed. Consequently, the number of training samples has been adjusted from 37,590 to 32,578, and the number of testing samples has increased from 4,296 to 5,012.

2. New Model Architecture
We modified the network architecture. In our latest testing, we observed that incorporating the Squeeze-and-Excitation block and a hybrid loss function combining time-domain and frequency-domain features yielded superior results. Consequently, we have made the corresponding adjustments.

3. Ablation Study Codes and check points
In the latest testing, we have refined the ablation study and submitted the corresponding codes. The specific details corresponding to the experiment IDs are presented in the table below.
| Exp. ID | SE block | Hybrid Loss | TFEM | DCT |
| :--- | :---: | :---: | :---: | :---: |
| exp 1 | × | × | ✓ | ✓ |
| exp 2 | ✓ | × | ✓ | ✓ |
| exp 4 | × | ✓ | ✓ | ✓ |
| exp 5 | ✓ | ✓ | ✓ | ✓ |
| exp 6 | ✓ | ✓ | × | ✓ |
| exp 7 | ✓ | ✓ | × | × |
| exp 8 | ✓ | ✓ | ✓ | x |

Additionally, in exp 9, we attempted to replace Discrete Cosine Transform (DCT) with Discrete Wavelet Transform (DWT).

4. Comparative Methods Codes
We have submitted the code used in the comparative experiments.

5. Host Codes for Wearable ECG Devices
We have provided the host codes for wearable ECG devices based on the AD8232 chip.


This repository provides the implementation of [TFCDiff: Robust ECG Denoising via Time-Frequency Complementary Diffusion](https://www.authorea.com/users/1003070/articles/1363493-tfcdiff-robust-ecg-denoising-via-time-frequency-complementary-diffusion).

`Abstract`
Ambulatory electrocardiogram (ECG) readings are prone to mixed noise from physical activities, including baseline wander (BW), muscle artifact (MA), and electrode motion artifact (EM). Developing a method to remove such complex noise and reconstruct high-fidelity signals is clinically valuable for diagnostic accuracy. However, denoising of multi-beat ECG segments remains understudied and poses technical challenges. To address this, we propose Time-Frequency Complementary Diffusion (TFCDiff), a novel approach that operates in the Discrete Cosine Transform (DCT) domain and uses the DCT coefficients of noisy signals as conditioning input. To refine waveform details, we incorporate Temporal Feature Enhancement Mechanism (TFEM) to reinforce temporal representations and preserve key physiological information. Comparative experiments on a synthesized dataset demonstrate that TFCDiff achieves state-of-the-art performance across five evaluation metrics. Furthermore, TFCDiff shows superior generalization on the unseen SimEMG Database, outperforming all benchmark models. Notably, TFCDiff processes raw 10-second sequences and maintains robustness under flexible random mixed noise (fRMN), enabling plug-and-play deployment in wearable ECG monitors for high-motion scenarios.

<div align="center">
  <img src="./images/TFCDiff_workflow.png" width="80%">
</div>
<div align="center">
  <img src="./images/Architecture.png" width="80%">
</div>

## Datasets
We use the following two datasets for training and intra-dataset testing:

1. QT Database: [QTDB](https://physionet.org/content/qtdb/1.0.0/)
   
2. MIT-BIH Noise Stress Test Database: [NSTDB](https://physionet.org/content/nstdb/1.0.0/)
   
We use the following dataset for inter-dataset testing:

3. SimEMG Database: [SimEMG](https://data.mendeley.com/datasets/yx5pb66hwz/1)

Unzip all the datasets and put them in the directory `data/`.

## Training
Please check `config/base.yaml` for the configuration first. The default configuration is

```
unet_config:
  in_channel: 2
  out_channel: 1
  inner_channel: 64
  channel_mults: [1, 2, 2, 2]
  attn_res: [250,]
  res_blocks: 2
  dropout: 0.0
  seq_len: 1000
  norm_groups: 16

train:
  epochs: 400
  batch_size: 128
  lr: 1.0e-3

diffusion:
  beta_start: 0.0001
  beta_end: 0.5
  num_steps: 50
  schedule: "quad"
```
Start training:
```
python -W ignore main.py --device cuda:0 --n_type=1
```

## Evaluation
Run evaluation code:
```
python complete_eval.py
```

## Denoising Your Own Dataset
Run denoising code:
```
python denoising.py
```

## Results
This table presents the overall comparison results of different methods for ECG denoising on the synthesized dataset. The noise level ranges from 0.2 to 2.

![image3](./images/Intradataset.png)

Visualization of the denoising results on the synthesized dataset.

![image4](./images/Synthesized.png)

This table presents the overall comparison results of different methods for ECG denoising on the SimEMG Database.

![image5](./images/Interdataset.png)

Visualization of the denoising results on the SimEMG Database.

![image6](./images/SimEMG.png)


