import shutil
import os

# 定义需要替换的文件列表
files_to_replace = [
    'unet.py',
    'main.py',
    'diffusion.py',
    'utils.py',
    os.path.join('Data_Preparation', 'data_preparation.py'),
    os.path.join('config', 'base.yaml')
]

# 源目录
source_dir = './backup/2026-04-14_13-41-28'

# 遍历需要替换的文件
for file_path in files_to_replace:
    # 源文件的完整路径
    source_file = os.path.join(source_dir, os.path.basename(file_path))
    # 目标文件的完整路径
    target_file = os.path.join('./', file_path)

    # 检查源文件是否存在
    if os.path.exists(source_file):
        # 检查目标文件所在的目录是否存在，如果不存在则创建
        target_dir = os.path.dirname(target_file)
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)

        try:
            # 复制源文件到目标文件，覆盖目标文件
            shutil.copy2(source_file, target_file)
            print(f"成功替换 {target_file}")
        except Exception as e:
            print(f"替换 {target_file} 时出错: {e}")
    else:
        print(f"源文件 {source_file} 不存在，无法替换。")