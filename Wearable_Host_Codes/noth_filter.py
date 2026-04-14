import numpy as np
from scipy import signal


class NotchFilter:
    """
    50Hz数字陷波器类，支持IIR巴特沃斯滤波器和梳状滤波器

    功能特点：
    - 支持任意采样率
    - 可选择滤波器类型
    - 支持连续数据流处理
    - 可重置滤波器状态
    - 提供频率响应分析

    使用方法：
    1. 初始化滤波器: filter = NotchFilter(fs, filter_type='butterworth')
    2. 处理数据: filtered_data = filter.process(data)
    3. (可选)重置状态: filter.reset()
    """

    def __init__(self, fs, filter_type='butterworth', notch_freq=50.0, quality_factor=30.0, harmonics=5):
        """
        初始化陷波滤波器

        参数:
        fs: 采样率 (Hz)
        filter_type: 滤波器类型 ('butterworth' 或 'comb')
        notch_freq: 陷波频率 (Hz), 默认为50Hz
        quality_factor: 品质因数(仅对巴特沃斯滤波器有效), 控制带宽
        harmonics: 谐波数量(仅对梳状滤波器有效)
        """
        self.fs = fs
        self.notch_freq = notch_freq
        self.filter_type = filter_type.lower()
        self.quality_factor = quality_factor
        self.harmonics = harmonics

        # 设计滤波器
        self.b, self.a = self._design_filter()

        # 初始化滤波器状态
        self._zi = None
        self.reset()

    def _design_filter(self):
        """根据参数设计滤波器"""
        if self.filter_type == 'butterworth':
            return self._design_butterworth_filter()
        elif self.filter_type == 'comb':
            return self._design_comb_filter()
        else:
            raise ValueError(f"未知滤波器类型: {self.filter_type}. 可用选项: 'butterworth', 'comb'")

    def _design_butterworth_filter(self):
        """设计巴特沃斯IIR陷波滤波器"""
        nyq = 0.5 * self.fs
        freq = self.notch_freq / nyq
        return signal.iirnotch(freq, self.quality_factor)

    def _design_comb_filter(self):
        """设计梳状陷波滤波器（适用于基频及其谐波）"""
        b_list, a_list = [], []

        for i in range(1, self.harmonics + 1):
            current_freq = self.notch_freq * i
            if current_freq >= self.fs / 2:
                break  # 超过奈奎斯特频率则跳过

            nyq = 0.5 * self.fs
            freq = current_freq / nyq
            b, a = signal.iirnotch(freq, self.quality_factor)
            b_list.append(b)
            a_list.append(a)

        # 级联所有滤波器
        b_comb, a_comb = np.array([1.0]), np.array([1.0])
        for bi, ai in zip(b_list, a_list):
            b_comb = np.convolve(b_comb, bi)
            a_comb = np.convolve(a_comb, ai)

        return b_comb, a_comb

    def reset(self):
        """重置滤波器状态"""
        self._zi = signal.lfilter_zi(self.b, self.a)

    def process_buffer(self, data, reset_state=False):
        """
        处理输入数据

        参数:
        data: 输入数据 (1D numpy数组)
        reset_state: 是否在处理前重置滤波器状态

        返回:
        滤波后的数据
        """
        if reset_state:
            self.reset()

        if len(data) == 0:
            return np.array([])

        # 使用滤波器状态实现连续处理
        filtered, self._zi = signal.lfilter(self.b, self.a, data, zi=self._zi)
        return filtered

    def freq_response(self, n_points=2000):
        """
        计算并返回滤波器的频率响应

        参数:
        n_points: 频率点数

        返回:
        freqs: 频率数组 (Hz)
        response: 复数频率响应
        """
        w, h = signal.freqz(self.b, self.a, worN=n_points, fs=self.fs)
        return w, h

    def plot_response(self, ax=None, max_freq=200, **kwargs):
        """
        绘制滤波器频率响应

        参数:
        ax: matplotlib轴对象 (可选)
        max_freq: 显示的最大频率 (Hz)
        kwargs: 传递给plot的其他参数
        """
        import matplotlib.pyplot as plt

        w, h = self.freq_response()
        mask = w <= max_freq

        if ax is None:
            plt.figure()
            ax = plt.gca()

        ax.plot(w[mask], 20 * np.log10(abs(h[mask])), **kwargs)
        ax.set_title(f'{self.filter_type.capitalize()} Notch Filter Frequency Response')
        ax.set_xlabel('Frequency [Hz]')
        ax.set_ylabel('Amplitude [dB]')
        ax.grid(True)
        ax.set_xlim(0, max_freq)
        ax.set_ylim(-60, 5)

        if ax is None:
            plt.tight_layout()

    def get_filter_coefficients(self):
        """获取滤波器系数 (b, a)"""
        return self.b, self.a


# 示例使用
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # 创建测试信号
    fs = 1000  # 采样率
    t = np.arange(0, 1.0, 1.0 / fs)
    test_signal = (np.sin(2 * np.pi * 10 * t) +  # 10Hz信号
                   0.5 * np.sin(2 * np.pi * 50 * t) +  # 50Hz干扰
                   0.3 * np.sin(2 * np.pi * 100 * t) +  # 100Hz干扰(二次谐波)
                   0.2 * np.sin(2 * np.pi * 150 * t))  # 150Hz干扰(三次谐波)

    # 创建滤波器实例
    butterworth_filter = NotchFilter(fs, filter_type='butterworth')
    comb_filter = NotchFilter(fs, filter_type='comb', harmonics=3)

    # 处理信号
    bw_filtered = butterworth_filter.process_buffer(test_signal)
    comb_filtered = comb_filter.process_buffer(test_signal)

    # 绘制结果
    plt.figure(figsize=(12, 8))

    # 频率响应
    plt.subplot(2, 1, 1)
    butterworth_filter.plot_response(label='Butterworth', linestyle='-')
    comb_filter.plot_response(label='Comb', linestyle='--')
    plt.legend()

    # 时域信号
    plt.subplot(2, 1, 2)
    plt.plot(t, test_signal, 'b-', alpha=0.5, label='Original')
    plt.plot(t, bw_filtered, 'r-', linewidth=1.5, label='Butterworth Filtered')
    plt.plot(t, comb_filtered, 'g-', linewidth=1.5, label='Comb Filtered')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude')
    plt.title('Filtering Results')
    plt.legend()
    plt.grid(True)
    plt.xlim(0, 0.1)  # 显示前0.1秒

    plt.tight_layout()
    plt.show()