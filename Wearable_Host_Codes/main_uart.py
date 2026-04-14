import asyncio
import csv
import sys
import threading

import datetime
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, QObject, QEventLoop, QTimer, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QComboBox, QPushButton, QCheckBox, QHBoxLayout, QMessageBox, QFileDialog, QSizePolicy, QSpacerItem
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
import qasync
import serial
from serial.tools import list_ports
from rt8232 import RT8232Data, RT8232Cmd

# 主线程
class MainWindow(QMainWindow, QObject):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.RtData = RT8232Data() # 数据缓存及解析
        self.serial_port = None # 串口对象

    def initUI(self):

        # 界面的标题
        self.setWindowTitle('RT8232模块-Python示例')

        # 界面的位置和大小的设置
        self.setGeometry(100, 100, 1800, 900)

        # 创建整体界面的水平布局
        hbox = QHBoxLayout()

        # 左侧垂直布局
        left_vbox = QVBoxLayout()
        left_vbox.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        left_vbox.addSpacerItem(QSpacerItem(120, 80, QSizePolicy.Fixed, QSizePolicy.Fixed))
        left_vbox.setSpacing(15)

        # 创建下拉框/信号类型
        self.port_combo = QComboBox(self)
        self.port_combo.addItems([])
        hbox1 = QHBoxLayout()
        hbox1.addWidget(QLabel("选择串口"))
        hbox1.addWidget(self.port_combo)

        self.btn_open = QPushButton('打开串口', self)
        self.btn_open.clicked.connect(self.open_serial)
        self.btn_start = QPushButton('开始采集', self)
        self.btn_start.clicked.connect(self.start_collect)
        self.btn_export_csv = QPushButton('导出数据', self)
        self.btn_export_csv.clicked.connect(self.export_csv)

        self.cb_filter_switch = QCheckBox("数字滤波器开关", self)
        self.cb_filter_switch.setChecked(True)
        self.cb_filter_switch.stateChanged.connect(self.filter_sw_update_status)

        left_vbox.addLayout(hbox1)
        left_vbox.addWidget(self.btn_open)
        left_vbox.addWidget(self.btn_start)
        left_vbox.addWidget(self.btn_export_csv)
        left_vbox.addWidget( self.cb_filter_switch)

        # 右侧垂直布局
        right_vbox = QVBoxLayout()

        # 创建一个绘图区域（QWidget）
        self.drawing_area = QWidget(self)
        self.drawing_area.setFixedSize(10, 10)
        left_vbox.addWidget(self.drawing_area)

        # 创建一个FigureCanvas来显示图像，修改为2行1列的子图
        self.fig, self.axes = plt.subplots(2, 1, layout='constrained')
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = NavigationToolbar(self.canvas, self)

        right_vbox.addWidget(self.canvas)
        right_vbox.addWidget(self.toolbar)

        hbox.addLayout(left_vbox)
        hbox.addLayout(right_vbox)

        # 创建一个中心小部件并设置布局
        central_widget = QWidget()
        central_widget.setLayout(hbox)
        self.setCentralWidget(central_widget)

        # 设置支持中文的字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
        plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

        # [修改] 设置图形和坐标轴
        # 子图1：原始ECG数据
        self.line_a_out, = self.axes[0].plot([], [], lw=1, color='blue', label='ECG')
        self.axes[0].grid(True)
        self.axes[0].legend(loc="upper right")
        self.axes[0].set_xlim(0, 300)
        # [修改] 原始数据保持固定量程
        self.axes[0].set_ylim(-4, 4)
        self.axes[0].set_ylabel("mV")
        self.axes[0].set_title('心电信号')

        # 子图2：AI去噪后的ECG数据
        self.line_ai_denoised, = self.axes[1].plot([], [], lw=1, color='green', label='AI Denoised')
        self.axes[1].grid(True)
        self.axes[1].legend(loc="upper right")
        self.axes[1].set_xlim(0, 300)
        self.axes[1].set_ylim(-0.5, 1.5) 
        self.axes[1].set_ylabel("mV (Normalized)")
        self.axes[1].set_xlabel("Samples")

        # 定时读取数据、处理数据、显示数据
        self.update_show_timer = QTimer(self)
        self.update_show_timer.timeout.connect(self.update_show)
        self.update_show_timer.start(100)

        # 更新串口列表
        self.refresh_ports()

    #  刷新可用的串口列表
    def refresh_ports(self):
        # 保存当前选择的串口
        current_selection = self.port_combo.currentText()
        self.port_combo.clear()

        # 获取所有可用串口
        ports = serial.tools.list_ports.comports()

        if not ports:
            self.port_combo.addItem("未检测到串口")
            self.port_combo.setEnabled(False)
            self.btn_open.setEnabled(False)
        else:
            for port in sorted(ports):
                self.port_combo.addItem(port.device, port.description)
            self.port_combo.setEnabled(True)
            self.btn_open.setEnabled(True)

            # 尝试恢复之前的选择
            if current_selection in [self.port_combo.itemText(i) for i in range(self.port_combo.count())]:
                self.port_combo.setCurrentText(current_selection)
            elif ports:  # 默认选择第一个串口
                self.port_combo.setCurrentIndex(0)

    # 打开或关闭串口
    def open_serial(self):
       if self.btn_open.text() == "打开串口":
            port_name = self.port_combo.currentText()
            # 打开串口，将波特率配置为256000，数据位为8，停止位为1，无校验位，读超时时间为0.5秒。
            try:
                self.serial_port = serial.Serial(port=port_name,
                                                baudrate=256000,
                                                bytesize=serial.EIGHTBITS,
                                                parity=serial.PARITY_NONE,
                                                stopbits=serial.STOPBITS_ONE,
                                                timeout=0.5)

                if self.serial_port.isOpen():  # 判断串口是否成功打开
                    self.btn_open.setText("关闭串口")
                    print("打开串口成功。")
                    print(self.serial_port.name)  # 输出串口号
                else:
                    self.btn_open.setText("打开串口")
                    self.serial_port = None
                    print("打开串口失败。")
            except Exception as e:
                QMessageBox.warning(self, "错误", f"无法打开串口: {e}")
                self.btn_open.setText("打开串口")
                self.serial_port = None
       else:
           self.btn_open.setText("打开串口")
           if self.serial_port and self.serial_port.isOpen():
               self.serial_port.close()
           self.serial_port = None

    # 开始采集
    def start_collect(self):
        if self.btn_start.text() == "开始采集":
            if self.serial_port is None or not self.serial_port.isOpen():
                QMessageBox.warning(self, "警告", "请先打开串口！")
                return
            self.btn_start.setText("停止采集")
            self.RtData.clear()
            self.serial_port.write(bytes(RT8232Cmd.start_collect_cmd()))
        else :
            self.btn_start.setText("开始采集")
            self.serial_port.write(bytes(RT8232Cmd.stop_collect_cmd()))

    # 更新显示
    def update_show(self):
        if self.serial_port is None:
            return

        # 获取串口已接收的数据
        n = self.serial_port.in_waiting
        if n > 0:
           # 读取串口数据
           com_data =  self.serial_port.read(n)
           # 解析数据
           self.RtData.parse_data(com_data)
           
           # [修改] 更新绘图
           if len(self.RtData.ecg_data) > 0:
              # 绘制原始数据
                self.update_graph(self.axes[0], self.line_a_out, self.RtData.ecg_data, 500)

                # 绘制AI去噪数据
                # [核心修改] 在这里传入原始数据的长度作为参考，用于零值填充
                with self.RtData.ai_lock:
                  ai_data = self.RtData.ecg_ai_denoised.copy()
                # print("AI Denoised Length: ", len(ai_data))
                self.update_graph(self.axes[1], self.line_ai_denoised, ai_data, 500, ref_len=len(self.RtData.ecg_data))

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()

    # [修改] 更新曲线
    def update_graph(self, axes, line, data, rate, ref_len=None):
        # 如果没有数据，直接返回
        if len(data) == 0:
            return

        # [修改] 零值填充逻辑
        # 只有当 ref_len 被提供且当前数据长度小于 ref_len 时才进行填充
        disp_len_source = len(data)
        if ref_len is not None and ref_len > disp_len_source:
            # np.pad 操作生成一个新数组，不改变原来的 data
            # 先转换为numpy数组方便操作
            np_data = np.array(data) if not isinstance(data, np.ndarray) else data
            # 在末尾填充 0，直到长度达到 ref_len
            data = np.pad(np_data, (0, ref_len - disp_len_source), 'constant', constant_values=np.nan)
        
        disp_length = rate * 30
        
        len_n = len(data)
        if len_n < disp_length:
            disp_length = len_n
        
        x0 = len_n - disp_length
        x1 = len_n
        
        y_data = data[x0:x1]
        x_data = np.linspace(x0, x1, disp_length)
        
        line.set_data(x_data, y_data)  # 更新曲线的数据
        axes.set_xlim(x0, x1)

    # [修改] 导出csv数据
    def export_csv(self):

        if len(self.RtData.ecg_data)  == 0 and len(self.RtData.ecg_ai_denoised) == 0 :
            QMessageBox.information(None, "提示", "数据缓存区为空，请先采集数据!", QMessageBox.Ok)
            return

        now = datetime.datetime.now()
        base_name = now.strftime("%Y%m%d_%H%M%S") + "_" + self.RtData.signal_type + "_" + str(self.RtData.sample_rate) + "sps"
        
        # 保存原始数据 CSV
        filepath_raw = ""
        filepath_ai = ""
        
        if len(self.RtData.ecg_data) > 0:
            filepath_raw, _ = QFileDialog.getSaveFileName(
                None,
                "保存原始ECG数据",
                base_name + "_raw.csv",
                "CSV文件 (*.csv);;所有文件 (*)"
            )

            if filepath_raw != "":
                headers = ["ECG"]
                with open(filepath_raw, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    writer.writerows([[f"{num:.3f}"] for num in self.RtData.ecg_data])

        # 保存AI去噪数据 CSV
        if len(self.RtData.ecg_ai_denoised) > 0:
            filepath_ai, _ = QFileDialog.getSaveFileName(
                None,
                "保存AI去噪ECG数据",
                base_name + "_ai_denoised.csv",
                "CSV文件 (*.csv);;所有文件 (*)"
            )

            if filepath_ai != "":
                headers = ["ECG_AI_Denoised"]
                with open(filepath_ai, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    writer.writerow(headers)
                    with self.RtData.ai_lock:
                        data_to_write = self.RtData.ecg_ai_denoised.copy()
                    writer.writerows([[f"{num:.3f}"] for num in data_to_write])
        
        if filepath_raw or filepath_ai:
            QMessageBox.information(None, "提示", "导出成功！", QMessageBox.Ok)

    def filter_sw_update_status(self):
        self.RtData.filter_switch =  self.cb_filter_switch.isChecked()

async def main():
    app = QApplication(sys.argv)
    # 使用 qasync 将 asyncio 事件循环与 PyQt 5 集成
    loop = qasync.QEventLoop(app)
    asyncio.set_event_loop(loop)

    # 窗体UI
    main_window = MainWindow()
    main_window.show()

    # 启动事件循环
    with loop:
        loop.run_forever()

if __name__ == '__main__':
    asyncio.run(main())