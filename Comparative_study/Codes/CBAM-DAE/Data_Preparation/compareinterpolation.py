from PIL import Image
import os

# 定义三个输入文件夹和一个输出文件夹
folder1 = r'C:\Users\PC\Desktop\AF_prediction\Step1_denoising\Data_Preparation\test_qt\noremoval'
folder2 = r'C:\Users\PC\Desktop\AF_prediction\Step1_denoising\Data_Preparation\test_qt\withremoval_v2_一阶'
folder3 = r'C:\Users\PC\Desktop\AF_prediction\Step1_denoising\Data_Preparation\test_qt\withremoval_v9'
output_folder = r'C:\Users\PC\Desktop\AF_prediction\Step1_denoising\Data_Preparation\test_qt\compare'

# 确保输出文件夹存在
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取第一个文件夹中的所有 PNG 文件
png_files = [f for f in os.listdir(folder1) if f.endswith('.png')]

for file in png_files:
    file_path1 = os.path.join(folder1, file)
    file_path2 = os.path.join(folder2, file)
    file_path3 = os.path.join(folder3, file)

    # 检查三个文件夹中是否都存在同名的 PNG 文件
    if os.path.exists(file_path1) and os.path.exists(file_path2) and os.path.exists(file_path3):
        # 打开三张图片
        img1 = Image.open(file_path1)
        img2 = Image.open(file_path2)
        img3 = Image.open(file_path3)

        # 获取三张图片的宽度和高度
        width1, height1 = img1.size
        width2, height2 = img2.size
        width3, height3 = img3.size

        # 确保三张图片的宽度相同
        if width1 == width2 == width3:
            # 计算合并后图片的总高度
            total_height = height1 + height2 + height3

            # 创建一个新的空白图片，宽度为三张图片的宽度，高度为三张图片高度之和
            merged_image = Image.new('RGB', (width1, total_height))

            # 将三张图片依次粘贴到新图片上
            merged_image.paste(img1, (0, 0))
            merged_image.paste(img2, (0, height1))
            merged_image.paste(img3, (0, height1 + height2))

            # 保存合并后的图片到输出文件夹
            output_path = os.path.join(output_folder, file)
            merged_image.save(output_path)

            print(f"合并并保存了 {file}")
        else:
            print(f"警告：{file} 的宽度不一致，无法合并。")
    else:
        print(f"警告：{file} 在某些文件夹中不存在，跳过。")

print("合并完成。")