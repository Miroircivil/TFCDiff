import struct
import numpy as np
from rt8232_filter import EcgFilter
import yaml
from diffusion import DDPM
from unet import UNet
import torch
import torch_dct as dct
import torch.nn.functional as F
from scipy.signal import resample_poly
import threading  # 新增：导入多线程库
import queue     # 新增：导入队列库

class RT8232Data:
    def __init__(self):
        self.rx_buff = []           # 数据包接收缓存区
        self.ecg_data = []          # 脉搏模拟信号（主线程实时更新）
        self.ecg_ai_denoised = []   # AI处理后的心电数据（后台线程异步更新）
        
        # self.ecg_ai_temp 被队列替代，用于线程间传递数据
        self.ai_task_queue = queue.Queue() 
        
        self.sample_rate = 500
        self.signal_type = "ECG"
        self.EcgFilter = EcgFilter()
        self.filter_switch = True

        # --- AI模型初始化相关配置 ---
        self.shots = 3
        self.eta = 3
        self.ai_initialized = False
        self.device = 'cuda:0'
        self.model = None

        # --- 滑动窗口配置 ---
        self.window_size_sec = 10
        self.stride_sec = 5
        self.overlap_sec = self.window_size_sec - self.stride_sec
        self.window_size_samples = 360 * self.window_size_sec
        self.stride_samples = 360 * self.stride_sec
        self.overlap_samples = 360 * self.overlap_sec
        self.pending_overlap_data = [] 
        self.has_pending_overlap = False

        # --- 线程与同步控制 ---
        self.ai_lock = threading.Lock() # 用于保护 self.ecg_ai_denoised 的读写
        self._stop_ai_thread = False   # 标志位，用于安全停止线程
        self.ai_thread = None          # 线程对象
        
        # 启动后台处理线程
        self._start_ai_worker_thread()

    def _start_ai_worker_thread(self):
        """启动后台AI处理线程"""
        self.ai_thread = threading.Thread(target=self._ai_worker_loop, daemon=True)
        self.ai_thread.start()

    def _ai_worker_loop(self):
        """后台线程的工作循环：不断检查队列并进行AI计算"""
        while not self._stop_ai_thread:
            try:
                raw_data_chunk = self.ai_task_queue.get(timeout=0.1)
                
                if not hasattr(self, '_worker_buffer'):
                    self._worker_buffer = []
                
                self._worker_buffer.extend(raw_data_chunk)
                self.ai_task_queue.task_done()
                
                # 检查是否攒够了数据
                if len(self._worker_buffer) >= self.window_size_samples:
                    # 切片取出10秒数据
                    segment = np.array(self._worker_buffer[:self.window_size_samples])
                    # 截断窗口步长
                    self._worker_buffer = self._worker_buffer[self.stride_samples:]

                    # 执行计算
                    self._process_ai_segment(segment)

            except queue.Empty:
                # 队列为空，继续循环
                continue
            except Exception as e:
                print(f"后台AI处理线程出错: {e}")

    def _init_ai_model(self):
        """初始化AI模型（线程安全）"""
        if self.ai_initialized:
            return

        # Load config
        path = "config/base.yaml"
        try:
            with open(path, "r") as f:
                config = yaml.safe_load(f)
        except FileNotFoundError:
            print("配置文件未找到，跳过AI初始化")
            return

        # Load model
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
        ).to(self.device)
        
        self.model = DDPM(base_model, config, self.device)
        output_path = "./check_points/model.pth"
        self.model.load_state_dict(torch.load(output_path, weights_only=False))
        self.model.eval()
        
        self.ai_initialized = True

    def _process_ai_segment(self, segment):
        """执行具体的AI推理逻辑（在后台线程中调用）"""
        try:
            # 懒初始化：确保模型已加载
            if not self.ai_initialized:
                self._init_ai_model()

            segment_tensor = torch.as_tensor(segment, dtype=torch.float32, device=self.device)  # 直接转为 GPU tensor

            # reshape 为 (1, 1, L) 以适配 interpolate
            X_ecg = segment_tensor.view(1, 1, -1) # [1, 1, 3600]

            # 重采样到 3600 点（线性插值）
            #segment_360 = F.interpolate(segment_5000, size=3600, mode='linear', align_corners=True)  # [1, 1, 3600]


            # DCT变换
            X_ecg_dct = dct.dct(X_ecg, norm='ortho')[:, :, :1000] / self.eta

            # 模型去噪
            # output = 0
            # for _ in range(self.shots):
            #     with torch.no_grad(): # 显式添加 no_grad 以节省显存
            #         output += self.model.denoising(X_ecg_dct)
            # output /= self.shots
            X_ecg_dct_shots = X_ecg_dct.repeat(self.shots, 1, 1)
            with torch.no_grad():
                output = self.model.denoising(X_ecg_dct_shots)
            output = output.mean(dim=0, keepdim=True)

            # 后处理
            output = F.pad(output, (0, 2600), mode='constant', value=0) * self.eta
            output = dct.idct(output, norm='ortho')
            #output = F.interpolate(output, size=self.sample_rate*10, mode='linear', align_corners=True)
            out_numpy = output.cpu().detach().numpy().flatten()

            final_output_chunk = []
            with self.ai_lock:
                if not self.has_pending_overlap:
                    final_output_chunk = out_numpy[:self.stride_samples]
                    self.has_pending_overlap = True
                    self.pending_overlap_data = out_numpy[self.stride_samples:]
                else:
                    current_overlap = out_numpy[:self.overlap_samples]
                    fused_overlap = (current_overlap + self.pending_overlap_data) / 2
                    out_numpy[:self.overlap_samples] = fused_overlap
                    final_output_chunk = out_numpy[:self.stride_samples]
                    self.pending_overlap_data = out_numpy[self.stride_samples:]

                final_output_chunk = final_output_chunk.tolist()
                if final_output_chunk:
                    self.ecg_ai_denoised.extend(final_output_chunk)
                
        except Exception as e:
            print(f"AI降噪推理出错: {e}")

    def set_signal_type(self, type):
        self.signal_type = type

    def clear(self):
        self.rx_buff = []
        self.ecg_data = []
        # 清空时的同步问题：如果有线程正在写入，清空会导致数据错乱
        with self.ai_lock:
            self.ecg_ai_denoised = []
            self.pending_overlap_data = []
            self.has_pending_overlap = False
        
        # 注意：这里我们无法直接清空 self._worker_buffer，因为它在子线程中
        # 一个简化的处理是重置它（如果对象存在）
        if hasattr(self, '_worker_buffer'):
            self._worker_buffer = []
            
        self.ai_task_queue.queue.clear() # 清空剩余任务
        self.EcgFilter = EcgFilter()

    def parse_data(self, data_in):
        """主线程调用的解析函数，极快返回，不阻塞"""
        self.rx_buff += data_in
        data = self.rx_buff

        while len(data) > 6:
            checkByte = data[1] ^ data[2]

            if (data[0] == 0xA5) and (data[3] == checkByte):
                framLen = data[1] + 3

                if framLen <= len(data):
                    if data[framLen - 1] == 0x5A:
                        self.frame_unpack(bytes(data[0:framLen]))
                        del data[0:framLen]
                    else:
                        data.pop(0)
                        # print("校验帧尾出错!") # 注释掉以免刷屏
                else:
                    break
            else:
                data.pop(0)
                # print("校验帧头出错!")

        self.rx_buff = data

    def frame_unpack(self, data):
        frameType = data[2]

        if frameType == RT8232Cmd.ADDRESS_SAMPLE_PAR:
            sample_rates = [125,250,500,1000,2000]
            self.sample_rate = sample_rates[data[4]]

        if frameType == RT8232Cmd.ADDRESS_START:
            data_num = int((data[1] - 2) / 2)
            temp = data[4:(4 + data_num * 2)]
            data_temp = struct.unpack("H" * (len(temp) // 2), temp)
            ecg = np.array(data_temp).astype(float)/10 

            if self.filter_switch:
                ecg = self.EcgFilter.process_buffer(self.sample_rate, ecg)
            
            #5是经验值保证数值幅度在1左右
            ecg = ((ecg - 2048) * 3300) / (4095 * 100 * 5)

            ecg = resample_poly(ecg, 360, self.sample_rate)
            
            # 1. 立即更新主数据列表（主线程）
            self.ecg_data.extend(ecg.tolist())

            # 2. 将原始数据放入队列，供后台线程处理
            # 放入的是 ecg (numpy array)
            try:
                self.ai_task_queue.put(ecg.tolist())
            except Exception as e:
                print(f"队列写入失败: {e}")

    def __del__(self):
        """析构时安全停止线程"""
        self._stop_ai_thread = True
        if self.ai_thread and self.ai_thread.is_alive():
            self.ai_thread.join(timeout=1.0)

# 指令集
class RT8232Cmd:
    ADDRESS_HW_VERSION = 0x01  #硬件版本
    ADDRESS_SOFTWARE = 0x02  #软件版本
    ADDRESS_DEVICE_NAME = 0x03  #设备名称
    ADDRESS_DEVICE_MAC = 0x04  # MAC地址
    ADDRESS_POWER = 0x05  #电量信息
    ADDRESS_RESET = 0x06 # 复位
    ADDRESS_SAMPLE_PAR = 0x10 # 采样参数
    ADDRESS_START = 0x11  #开始/停止采集指令

    @staticmethod
    def cmd_data_pack(addr, is_write, data):

        cmd = []

        # 帧头
        cmd.append(0xAA)
        # 帧长度
        cmd.append(len(data) + 3)
        if is_write:
            cmd.append(0x80)  # 写指令
        else:
            cmd.append(0x81)  # 读指令
        # 地址
        cmd.append(addr)

        # 数据内容
        cmd += data

        # 校验码（待定）
        cmd.append(0x00)

        # 帧尾
        cmd.append(0xBB)

        # 计算校验码
        xor = 0x00
        for b in cmd[1:]:
            xor ^= b
        cmd[len(cmd) - 2] = xor
        return cmd

    # 开始采集指令
    @staticmethod
    def start_collect_cmd():
        data = []
        data.append(0x01)
        return RT8232Cmd.cmd_data_pack(RT8232Cmd.ADDRESS_START, True, data)

    # 停止采集指令
    @staticmethod
    def stop_collect_cmd():
        data = []
        data.append(0x00)
        return RT8232Cmd.cmd_data_pack(RT8232Cmd.ADDRESS_START, True, data)