import wfdb
import numpy as np
import matplotlib.pyplot as plt
import os

# 心电记录的路径
QTpath = 'data/qt-database-1.0.0/'
record_name = 'sel30'
record_path = os.path.join(QTpath, record_name)

# 读取心电记录
signal, fields = wfdb.rdsamp(record_path)
# 提取第一个通道的数据
auxSig1 = signal[:, 0]
# 提取第二个通道的数据
auxSig2 = signal[:, 1]

# 读取第一个通道的标注
ann_pu0 = wfdb.rdann(record_path, 'pu0')
symbols_pu0 = ann_pu0.symbol
samples_pu0 = ann_pu0.sample
print(f"symbols_pu0 的类型是: {type(symbols_pu0)}")
print(f"samples_pu0 的类型是: {type(samples_pu0)}")

# 读取第二个通道的标注
ann_pu1 = wfdb.rdann(record_path, 'pu1')
symbols_pu1 = ann_pu1.symbol
samples_pu1 = ann_pu1.sample

# 输出目录
output_dir = r'C:\Users\PC\Desktop\AF_prediction\Step1_denoising\Data_Preparation\annotation'
os.makedirs(output_dir, exist_ok=True)

# 裁剪长度
clip_length = 2500

# 计算需要裁剪的片段数量
num_clips = len(auxSig1) // clip_length

for i in range(num_clips):
    start_idx = i * clip_length
    end_idx = start_idx + clip_length

    # 裁剪通道数据
    clipped_auxSig1 = auxSig1[start_idx:end_idx]
    clipped_auxSig2 = auxSig2[start_idx:end_idx]

    # 找出当前片段内的标注
    pu0_indices_in_clip = np.where((samples_pu0 >= start_idx) & (samples_pu0 < end_idx))[0]
    pu1_indices_in_clip = np.where((samples_pu1 >= start_idx) & (samples_pu1 < end_idx))[0]

    # 计算标注在裁剪后数据中的相对位置
    relative_samples_pu0 = samples_pu0[pu0_indices_in_clip] - start_idx
    relative_samples_pu1 = samples_pu1[pu1_indices_in_clip] - start_idx

    # 创建一个包含上下两个子图的图形
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # 绘制第一个通道的数据和标注
    ax1.plot(clipped_auxSig1)
    for j, sample in enumerate(relative_samples_pu0):
        ax1.text(sample, clipped_auxSig1[sample], symbols_pu0[pu0_indices_in_clip[j]], color='red')
    ax1.set_title('Channel 1')

    # 绘制第二个通道的数据和标注
    ax2.plot(clipped_auxSig2)
    for j, sample in enumerate(relative_samples_pu1):
        ax2.text(sample, clipped_auxSig2[sample], symbols_pu1[pu1_indices_in_clip[j]], color='red')
    ax2.set_title('Channel 2')

    # 调整子图之间的间距
    plt.tight_layout()

    # 保存图形
    save_path = os.path.join(output_dir, f'{record_name}_clip_{i}.png')
    plt.savefig(save_path)
    plt.close()

print(f"所有图形已保存到 {output_dir}")
