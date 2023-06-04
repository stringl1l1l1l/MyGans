import os
import shutil
from PIL import Image
from torchvision import transforms
import torch


def process_image(image_path, new_path):
    image = Image.open(image_path)
    if image.mode == 'RGBA':
        # # 创建一个纯白色的背景图像
        # background = Image.new('RGBA', image.size, (255, 255, 255))
        # # 将原始图像粘贴到背景图像上
        # background.paste(image, mask=image)
        # # 将图像模式更改为RGB
        # image = background.convert('RGB')
        image = image.convert('RGB')
    if image.mode == 'P':
        image = image.convert('RGB')
    # 统一图片分辨率
    target_resolution = (256, 256)
    image = image.resize(target_resolution, Image.BILINEAR)
    # 保存更改后的图像
    image.save(new_path)


def raw2jpg():
    data_dir = './raw_data'
    new_dir = './jpg_data'
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for tag in ["/pos", "/neg"]:
        if not os.path.exists(new_dir + tag):
            os.makedirs(new_dir + tag)
    for tag in ["/pos", "/neg"]:
        index = 1
        for filename in os.listdir(data_dir + tag):
            old_path = os.path.join(data_dir + tag, filename)
            new_path = os.path.join(new_dir + tag, f'{index}.jpg')
            # if filename.endswith('.jpg'):
            #     shutil.copy2(old_path, new_path)
            # else:
            process_image(old_path, new_path)
            index += 1
            print(new_path)


def jpg2tensor():
    jpg_data_dir = './jpg_data'
    tensor_data_dir = './tensor_data'
    if not os.path.exists(tensor_data_dir):
        os.makedirs(tensor_data_dir)
    for tag in ["/pos", "/neg"]:
        if not os.path.exists(tensor_data_dir + tag):
            os.makedirs(tensor_data_dir + tag)

    transform = transforms.ToTensor()
    transform = transforms.Normalize(
        mean=[0.5, 0.5, 0.5],  # 取决于数据集
        std=[0.5, 0.5, 0.5]
    )
    for tag in ["/pos", "/neg"]:
        for filename in os.listdir(jpg_data_dir + tag):
            if filename.endswith('.jpg'):
                image_path = os.path.join(jpg_data_dir + tag, filename)
                image = Image.open(image_path)
                tensor = transform(image)
                if (tensor.shape[0] != 3):
                    tensor = tensor[:3, :, :]
                tensor_path = os.path.join(tensor_data_dir + tag, f'{filename}.pt')
                torch.save(tensor, tensor_path)
                print(tensor_path)


raw2jpg()
jpg2tensor()
