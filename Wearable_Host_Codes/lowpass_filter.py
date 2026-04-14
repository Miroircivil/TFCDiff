import numpy as np
from scipy.signal import butter, lfilter

class LowPassFilter:
    def __init__(self, cutoff_freq,sample_rate,  order=5):
        """
        低通滤波器初始化

        参数:
            sample_rate (float): 采样率(Hz)
            cutoff_freq (float): 截止频率(Hz)
            order (int): 滤波器阶数，默认为5
        """
        self.sample_rate = sample_rate
        self.cutoff_freq = cutoff_freq
        self.order = order

        # 设计Butterworth低通滤波器系数
        nyquist = 0.5 * sample_rate
        normal_cutoff = cutoff_freq / nyquist
        self.b, self.a = butter(order, normal_cutoff, btype='low', analog=False)

        # 初始化滤波器状态
        self.zi = np.zeros(max(len(self.a), len(self.b)) - 1)

    def process(self, input_data):
        """
        处理输入数据

        参数:
            input_data (float or array-like): 输入数据点或数组

        返回:
            float or ndarray: 滤波后的输出数据
        """
        # 确保输入是numpy数组
        input_data = np.asarray(input_data)

        # 应用滤波器
        if input_data.ndim == 0:  # 标量输入
            input_data = np.array([input_data])
            output, self.zi = lfilter(self.b, self.a, input_data, zi=self.zi)
            return output[0]
        else:  # 数组输入
            output, self.zi = lfilter(self.b, self.a, input_data, zi=self.zi)
            return output


# 使用示例
if __name__ == "__main__":
    # 创建滤波器实例: 采样率1000Hz, 截止频率50Hz
    filter = LowPassFilter(sample_rate=1000.0, cutoff_freq=50.0)

    # 测试标量输入
    print("Testing scalar input:")
    for i in range(5):
        print(f"Input: {i}, Output: {filter.process(i)}")

    # 测试数组输入
    print("\nTesting array input:")
    test_array = np.array([1, 2, 3, 4, 5])
    print(f"Input array: {test_array}")
    print(f"Filtered array: {filter.process(test_array)}")

    # 模拟信号处理
    print("\nSimulating signal processing:")
    t = np.linspace(0, 1, 1000)
    input_signal = np.sin(2 * np.pi * 10 * t)  # 10Hz正弦波
    noise = 0.5 * np.sin(2 * np.pi * 200 * t)  # 200Hz噪声
    noisy_signal = input_signal + noise

    # 重置滤波器
    filter = LowPassFilter(sample_rate=1000.0, cutoff_freq=50.0)

    # 处理整个数组
    filtered_signal = filter.process(noisy_signal)

    # 绘制结果(需要matplotlib)
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        plt.plot(t, noisy_signal, label='Noisy Signal')
        plt.plot(t, filtered_signal, label='Filtered Signal', linewidth=2)
        plt.plot(t, input_signal, label='Original Signal', linestyle='--')
        plt.xlabel('Time [s]')
        plt.ylabel('Amplitude')
        plt.title('Low Pass Filtering ({}Hz cutoff)'.format(filter.cutoff_freq))
        plt.legend()
        plt.grid()
        plt.show()
    except ImportError:
        print("Matplotlib not available, skipping plot")
        print("First 10 filtered values:", filtered_signal[:10])