import numpy as np
from scipy.signal import butter, sosfilt_zi


class EnhancedHighPassFilter:
    def __init__(self, cutoff_freq, sample_rate, order=4):
        """
        增强版高通滤波器类，支持单点和批量处理

        参数:
            cutoff_freq (float): 截止频率(Hz)
            sample_rate (float): 采样率(Hz)
            order (int): 滤波器阶数(必须为偶数)

        异常:
            ValueError: 如果参数无效
        """
        # 参数验证
        if not isinstance(order, int) or order <= 0 or order % 2 != 0:
            raise ValueError("阶数必须是正整数且为偶数")
        if sample_rate <= 0:
            raise ValueError("采样率必须大于0")
        if cutoff_freq <= 0 or cutoff_freq >= sample_rate / 2:
            raise ValueError("截止频率必须在0和奈奎斯特频率之间")

        self.cutoff_freq = cutoff_freq
        self.sample_rate = sample_rate
        self.order = order

        # 设计滤波器
        self._design_filter()
        self.reset()

    def _design_filter(self):
        """设计滤波器系数"""
        nyquist = 0.5 * self.sample_rate
        normal_cutoff = self.cutoff_freq / nyquist

        # 设计巴特沃斯高通滤波器(二阶节形式)
        self.sos = butter(
            N=self.order,
            Wn=normal_cutoff,
            btype='high',
            analog=False,
            output='sos'
        )

        # 预计算增益补偿因子
        self.scale = np.prod(self.sos[:, 0])  # 所有b0系数的乘积

    def process_sample(self, sample):
        """
        处理单个采样点(实时处理)

        参数:
            sample (float): 输入采样值

        返回:
            float: 滤波后的输出值
        """
        output, self.zi = self._sosfilt_single(self.sos, np.array([sample]) * self.scale, self.zi)
        return output[0]

    def process_buffer(self, buffer):
        """
        处理一段采样数据(批量处理，效率更高)

        参数:
            buffer (np.ndarray): 输入采样数组

        返回:
            np.ndarray: 滤波后的输出数组
        """
        if not isinstance(buffer, np.ndarray):
            buffer = np.array(buffer, dtype=np.float64)

        # 应用增益补偿并滤波
        output, self.zi = self._sosfilt_single(self.sos, buffer * self.scale, self.zi)
        return output

    def reset(self):
        """重置滤波器状态"""
        self.zi = sosfilt_zi(self.sos) * 0  # 初始化为零状态

    @staticmethod
    def _sosfilt_single(sos, x, zi):
        """
        优化的单点/批量SOS滤波器实现

        参数:
            sos: 二阶节系数
            x: 输入信号(可以是单点或数组)
            zi: 滤波器状态

        返回:
            (输出数组, 更新后的状态)
        """
        x = np.asarray(x, dtype=np.float64)
        output = np.zeros_like(x)

        for i in range(sos.shape[0]):  # 遍历每个二阶节
            b0, b1, b2, a0, a1, a2 = sos[i, :]
            # 归一化系数
            b0 /= a0
            b1 /= a0
            b2 /= a0
            a1 /= a0
            a2 /= a0

            for n in range(x.shape[0]):  # 处理每个采样点
                # 直接形式II实现
                input_val = x[n] - a1 * zi[i, 0] - a2 * zi[i, 1]
                output_val = b0 * input_val + b1 * zi[i, 0] + b2 * zi[i, 1]

                # 更新状态
                zi[i, 1] = zi[i, 0]
                zi[i, 0] = input_val

                x[n] = output_val  # 传递到下一个二阶节

        output[:] = x[:]
        return output, zi

    def __str__(self):
        return (f"EnhancedHighPassFilter: "
                f"cutoff={self.cutoff_freq}Hz, "
                f"sample_rate={self.sample_rate}Hz, "
                f"order={self.order}")


# 使用示例
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from time import time

    # 测试配置
    sample_rate = 1000  # 1kHz采样率
    cutoff_freq = 50  # 50Hz截止频率
    duration = 1.0  # 1秒持续时间
    order = 6  # 6阶滤波器

    # 生成测试信号
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    signal = (np.sin(2 * np.pi * 10 * t) +  # 10Hz低频
              0.5 * np.sin(2 * np.pi * 100 * t) +  # 100Hz高频
              0.1 * np.random.normal(size=len(t)))  # 添加噪声

    # 创建滤波器实例
    hp_filter = EnhancedHighPassFilter(cutoff_freq, sample_rate, order)
    print(hp_filter)

    # 测试单点处理模式
    print("\n测试单点处理模式...")
    start_time = time()
    filtered_point_by_point = np.zeros_like(signal)
    for i, sample in enumerate(signal):
        filtered_point_by_point[i] = hp_filter.process_sample(sample)
    point_time = time() - start_time

    # 重置滤波器
    hp_filter.reset()

    # 测试批量处理模式
    print("测试批量处理模式...")
    start_time = time()
    filtered_batch = hp_filter.process_buffer(signal)
    batch_time = time() - start_time

    # 性能比较
    print(f"\n性能比较:")
    print(f"单点处理: {point_time * 1000:.2f} ms")
    print(f"批量处理: {batch_time * 1000:.2f} ms")
    print(f"批量处理比单点处理快 {point_time / batch_time:.1f} 倍")

    # 绘制结果
    plt.figure(figsize=(12, 8))
    plt.plot(t, signal, 'k-', alpha=0.3, label='原始信号 (10Hz + 100Hz + 噪声)')
    plt.plot(t, filtered_point_by_point, 'b-', label='单点处理结果')
    plt.plot(t, filtered_batch, 'r--', label='批量处理结果')
    plt.xlabel('时间 [秒]')
    plt.ylabel('振幅')
    plt.title(f'高通滤波器效果比较 (截止频率={cutoff_freq}Hz, 阶数={order})')
    plt.legend()
    plt.grid()
    plt.show()