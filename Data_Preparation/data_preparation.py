import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB
import os
import torch

def generate_noise(noise, samples):
    # ensure there is three types of noise
    assert noise.shape[0] == 3
    
    # Randomly crop
    noise1_start = np.random.randint(0, len(noise[0]) - samples + 1)
    noise1 = noise[0][noise1_start:noise1_start + samples]
    
    noise2_start = np.random.randint(0, len(noise[1]) - samples + 1)
    noise2 = noise[1][noise2_start:noise2_start + samples]
    
    noise3_start = np.random.randint(0, len(noise[2]) - samples + 1)
    noise3 = noise[2][noise3_start:noise3_start + samples]

    a1 = np.random.randint(0, 101)
    a2 = np.random.randint(0, 101)

    noise = a1 * noise1 + (100 - a1) * (a2 * noise2 + (100 - a2) * noise3)

    return noise / 100

def Data_Preparation(noise_version=1):

    print('Getting the Data ready ... ')

    # The seed is used to ensure the ECG always have the same contamination level
    # this enhance reproducibility
    seed = 1234
    np.random.seed(seed=seed)

    if not os.path.exists('data/QTDatabase.pkl'):
        Prepare_QTDatabase.prepare()
    print("QTDatabase.pkl has existed, loading......")
    if not os.path.exists('data/NoiseBWL.pkl'):
        Prepare_NSTDB.prepare()
    print("NoiseBWL.pkl has existed, loading......")

    # Load QT Database
    with open('data/QTDatabase.pkl', 'rb') as input:
        qtdb = pickle.load(input)

    # Load NSTDB
    with open('data/NoiseBWL.pkl', 'rb') as input:
        nstdb = pickle.load(input)

    #####################################
    # NSTDB
    #####################################

    [bw_signals, em_signals, ma_signals] = nstdb
    bw_signals = np.array(bw_signals)
    em_signals = np.array(em_signals)
    ma_signals = np.array(ma_signals)
    # split
    # bw
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]
    # em
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]
    # ma
    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]  
    

    #####################################
    # Data split
    #####################################

    if noise_version == 1:
        noise_test = np.vstack((bw_noise_channel2_b, em_noise_channel2_b, ma_noise_channel2_b))
        noise_train = np.vstack((bw_noise_channel1_a, em_noise_channel1_a, ma_noise_channel1_a))
    elif noise_version == 2:
        noise_test = np.vstack((bw_noise_channel1_b, em_noise_channel1_b, ma_noise_channel1_b))
        noise_train = np.vstack((bw_noise_channel2_a, em_noise_channel2_a, ma_noise_channel2_a))
    elif noise_version == 3:
        noise_test = np.vstack((bw_noise_channel1_a, em_noise_channel1_a, ma_noise_channel1_a))
        noise_train = np.vstack((bw_noise_channel2_b, em_noise_channel2_b, ma_noise_channel2_b))
    elif noise_version == 4:
        noise_test = np.vstack((bw_noise_channel2_a, em_noise_channel2_a, ma_noise_channel2_a))
        noise_train = np.vstack((bw_noise_channel1_b, em_noise_channel1_b, ma_noise_channel1_b))
    else:
        raise Exception("Sorry, noise_version should be 1 ~ 4")

    #####################################
    # QTDatabase
    #####################################

    # ground truth
    signals_train = []
    signals_test = []

    test_set = ['sel123',  # Record from MIT-BIH Arrhythmia Database
                'sel233',

                'sel302',  # Record from MIT-BIH ST Change Database
                'sel307',

                'sel820',  # Record from MIT-BIH Supraventricular Arrhythmia Database
                'sel853',

                'sel16420',  # Record from MIT-BIH Normal Sinus Rhythm Database
                'sel16795',

                'sele0106',  # Record from European ST-T Database
                'sele0121',

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',
                ]

    skip_signals = 0
    # fs = 360Hz, so duration = 10s
    samples = 3600

    # s_np is a 10-s segment cropped from a recording, signal_name is the ID of the recording
    for signal_name in qtdb.keys():
        for s in qtdb[signal_name]:
            s_np = np.array(s)
            # sometimes you might forget you have changed the length in data preparation as I did.
            if s_np.shape[0] != samples:
                skip_signals += 1
                continue
            signals_test.append(s_np) if signal_name in test_set else signals_train.append(s_np)
    
    # sn means signals with noise
    sn_train = []
    sn_test = []
    
    # Adding noise to train
    # rnd ~ [0.2, 2]
    rnd_train = np.random.randint(low=20, high=200, size=len(signals_train)) / 100
    for _ in range(1): # in case you need to augment the data
        for i in range(len(signals_train)):
            noise = generate_noise(noise_train, samples)
            signal_max_value = np.max(signals_train[i]) - np.min(signals_train[i])
            noise_max_value = np.max(noise) - np.min(noise)
            ratio = signal_max_value / noise_max_value
            alpha = rnd_train[i] * ratio
            signal_noise = signals_train[i] + alpha * noise
            sn_train.append(signal_noise)


    # Adding noise to test
    rnd_test = np.random.randint(low=20, high=200, size=len(signals_test)) / 100

    # Saving the random array so we can use it on the amplitude segmentation tables
    np.save('rnd_test.npy', rnd_test)
    print('rnd_test shape: ' + str(rnd_test.shape))
    for i in range(len(signals_test)):
        noise = generate_noise(noise_test, samples)
        signal_max_value = np.max(signals_test[i]) - np.min(signals_test[i])
        noise_max_value = np.max(noise) - np.min(noise)
        ratio = signal_max_value / noise_max_value
        alpha = rnd_test[i] * ratio
        signal_noise = signals_test[i] + alpha * noise
        sn_test.append(signal_noise)

    X_train = torch.unsqueeze(torch.FloatTensor(np.array(sn_train)), dim=-1)
    y_train = torch.unsqueeze(torch.FloatTensor(np.array(signals_train)), dim=-1)
    X_test = torch.unsqueeze(torch.FloatTensor(np.array(sn_test)), dim=-1)
    y_test = torch.unsqueeze(torch.FloatTensor(np.array(signals_test)), dim=-1)

    # check
    print(f"shape of X_test: {X_test.shape}")
    print(f"shape of y_test: {y_test.shape}")
    print(f"shape of X_train: {X_train.shape}")
    print(f"shape of y_train: {y_train.shape}")

    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset

#debug

#[a, b, c, d] = Data_Preparation(1)
