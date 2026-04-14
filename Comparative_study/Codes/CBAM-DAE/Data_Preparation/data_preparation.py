import numpy as np
import _pickle as pickle
from Data_Preparation import Prepare_QTDatabase, Prepare_NSTDB
import os
import torch_dct as dct
import torch

# def normalize(signals):
#     min_val = np.min(signals, axis=0)
#     max_val = np.max(signals, axis=0)
#     normalized_signals = (signals - min_val) / (max_val - min_val)
#     return normalized_signals

def generate_noise(noise, samples):
    # 确保第一维长度为 3
    assert noise.shape[0] == 3
    # 检查每个噪声信号的长度是否足够进行采样
    for i in range(3):
        assert len(noise[i]) >= samples, f"第 {i + 1} 个噪声信号的长度小于采样长度 {samples}"
    
    # 对三个噪声信号分别进行随机采样，得到长度为 samples 的连续部分
    noise1_start = np.random.randint(0, len(noise[0]) - samples + 1)
    noise1 = noise[0][noise1_start:noise1_start + samples]
    
    noise2_start = np.random.randint(0, len(noise[1]) - samples + 1)
    noise2 = noise[1][noise2_start:noise2_start + samples]
    
    noise3_start = np.random.randint(0, len(noise[2]) - samples + 1)
    noise3 = noise[2][noise3_start:noise3_start + samples]

    # 随机生成两个 0 到 100 的整数
    a1 = np.random.randint(0, 101)
    a2 = np.random.randint(0, 101)

    # 计算 noise
    noise = a1 * noise1 + (100 - a1) * (a2 * noise2 + (100 - a2) * noise3)

    # 返回 noise 除以 100 的结果
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
        # dict {register_name: signals_list}
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
    # 将每种噪声、每个通道都划分成前后两部分
    # bw 基线漂移
    bw_noise_channel1_a = bw_signals[0:int(bw_signals.shape[0]/2), 0]
    bw_noise_channel1_b = bw_signals[int(bw_signals.shape[0]/2):-1, 0]
    bw_noise_channel2_a = bw_signals[0:int(bw_signals.shape[0]/2), 1]
    bw_noise_channel2_b = bw_signals[int(bw_signals.shape[0]/2):-1, 1]
    # em 电极运动
    em_noise_channel1_a = em_signals[0:int(em_signals.shape[0]/2), 0]
    em_noise_channel1_b = em_signals[int(em_signals.shape[0]/2):-1, 0]
    em_noise_channel2_a = em_signals[0:int(em_signals.shape[0]/2), 1]
    em_noise_channel2_b = em_signals[int(em_signals.shape[0]/2):-1, 1]
    # ma 肌电干扰
    ma_noise_channel1_a = ma_signals[0:int(ma_signals.shape[0]/2), 0]
    ma_noise_channel1_b = ma_signals[int(ma_signals.shape[0]/2):-1, 0]
    ma_noise_channel2_a = ma_signals[0:int(ma_signals.shape[0]/2), 1]
    ma_noise_channel2_b = ma_signals[int(ma_signals.shape[0]/2):-1, 1]  
    

    #####################################
    # Data split 3和4为了严谨加上的，但保存路径可能还没定义，需要检查main_exp等代码，暂不可用
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

    # 这是干净信号的数组
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
                'sele0121'

                'sel32',  # Record from ``sudden death'' patients from BIH
                'sel49',

                'sel14046',  # Record from MIT-BIH Long-Term ECG Database
                'sel15814',
                ]

    skip_signals = 0
    samples = 3600
    #这里代码的设计有点冗余，用for signal_name in qtdb.keys()会更好，后期可以修改
    #这个代码的目的是生成干净信号的数组 s_np是合要求的一个信号，signal_name是心电记录的register_name
    qtdb_keys = list(qtdb.keys())
    # debug用：
    # print(f"qtdb_keys: {qtdb_keys}")
    
    for i in range(len(qtdb_keys)):
        signal_name = qtdb_keys[i]
        
        for s in qtdb[signal_name]:
            s_np = np.array(s)
            #print(f"signal_name: {signal_name}, s_np.shape: {s_np.shape[0]}")
            
            # 检查形状，按理说不应该有问题
            if s_np.shape[0] != samples:
                skip_signals += 1
                continue

            if signal_name in test_set:
                signals_test.append(s_np)
            else:
                signals_train.append(s_np)
    
    # 这是混杂了噪声的数组
    sn_train = []
    sn_test = []

    noise_index = 0
    # # 调试信息
    # print(f"Length of noise_train: {len(noise_train)}")
    # print(f"Length of noise_test: {len(noise_test)}")
    # print(f"Length of signals_train: {len(signals_train)}")
    # print(f"Length of signals_test: {len(signals_test)}")
    
    # Adding noise to train
    rnd_train = np.random.randint(low=20, high=200, size=len(signals_train)) / 100
    for _ in range(1):
        for i in range(len(signals_train)):
            noise = generate_noise(noise_train, samples)
            signal_max_value = np.max(signals_train[i]) - np.min(signals_train[i])
            noise_max_value = np.max(noise) - np.min(noise)
            ratio = signal_max_value / noise_max_value
            alpha = rnd_train[i] * ratio
            signal_noise = signals_train[i] + alpha * noise
            # signal_noise = normalize(signal_noise)
            sn_train.append(signal_noise)


    # Adding noise to test
    noise_index = 0
    rnd_test = np.random.randint(low=20, high=200, size=len(signals_test)) / 100
    #rnd_test = np.random.randint(low=150, high=200, size=len(signals_test)) / 100

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
        # signal_noise = normalize(signal_noise)
        sn_test.append(signal_noise)

    
    X_train = np.array(sn_train)[:,:3584]
    y_train = np.array(signals_train)[:,:3584]

    X_test = np.array(sn_test)[:,:3584]
    y_test = np.array(signals_test)[:,:3584]
    '''
    
    X_train = np.array(sn_train)
    
    y_train = np.array(signals_train)
    
    X_test = np.array(sn_test)
    y_test = np.array(signals_test)
    '''

    X_train = np.expand_dims(X_train, axis=-1)
    y_train = np.expand_dims(y_train, axis=-1)

    X_test = np.expand_dims(X_test, axis=-1)
    y_test = np.expand_dims(y_test, axis=-1)

    # debug用：
    print(f"X_test 的维度: {X_test.shape}")
    print(f"y_test 的维度: {y_test.shape}")
    print(f"X_train 的维度: {X_train.shape}")
    print(f"y_train 的维度: {y_train.shape}")

    #导出数据用
    # save_folder = r"C:\Users\PC\Desktop\AF_prediction\Score-based-ECG-Denoising-main\npdataset"
    # np.save(os.path.join(save_folder, 'X_train.npy'), X_train)
    # np.save(os.path.join(save_folder, 'y_train.npy'), y_train)
    # np.save(os.path.join(save_folder, 'X_test.npy'), X_test)
    # np.save(os.path.join(save_folder, 'y_test.npy'), y_test)


    Dataset = [X_train, y_train, X_test, y_test]

    print('Dataset ready to use.')

    return Dataset

#调试用
#[a, b, c, d] = Data_Preparation(1)