from torch.utils.data import Dataset
import os
import glob
import shutil
import numpy as np
import torch
from PIL import Image
import torchvision.transforms as transforms


class TinyImageNetDataset(Dataset):
    def __init__(self, src_path, input_shape, class_num, transform):
        folder_list = os.listdir(src_path)
        self.input_shape = input_shape
        self.class_num = class_num
        self.transform = transform
        self.x, self.y = [], []

        # 클래스 종류 리스트업
        with open("./tiny-imagenet-200/wnids_10class.txt", "r") as f:
            cls_list = f.readlines()
        cls_list = [cls_object.replace("\n", "") for cls_object in cls_list]

        for _, folder in enumerate(folder_list):
            imgs = glob.glob(src_path+folder+"/*.JPEG")
            if folder == ".DS_Store": continue
            cls = cls_list.index(folder)
            self.x += imgs
            self.y += list(np.full((len(imgs)), fill_value=cls))

    def __getitem__(self, idx):
        img = Image.open(self.x[idx])
        # 데이터셋에 잘못 들어간 흑백이미지 처리
        img = img.convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = torch.tensor(self.y[idx])
        return img, label

    def __len__(self):
        return len(self.x)


# train 9 test 1 비율로 데이터 분할
def load_data(root, split_ratio, train_dst_path, test_dst_path):
    if os.path.exists(train_dst_path) and os.path.exists(test_dst_path):
        return
    else:
        os.mkdir(train_dst_path)
        os.mkdir(test_dst_path)
    folders = os.listdir(os.path.join(root, 'train'))
    for folder in folders:
        os.mkdir(train_dst_path + folder)
        os.mkdir(test_dst_path + folder)
        img_paths = glob.glob(root + 'train_10class/' + folder + '/images/*.JPEG')
        img_len = len(img_paths)
        train_index = int(img_len * split_ratio)
        train_set = img_paths[:train_index]
        test_set = img_paths[train_index:img_len]
        for train in train_set:
            shutil.copy2(train, train_dst_path + folder)
        for test in test_set:
            shutil.copy2(test, test_dst_path + folder)


if __name__ == "__main__":
    # root = './tiny-imagenet-200/'
    # train_dst_path = "./train/"
    # test_dst_path = "./test/"
    # load_data(root, 0.9, train_dst_path, test_dst_path)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensodr()
    ])
    input_shape = (224, 224, 3)
    class_num = 10
    train_ds = TinyImageNetDataset(src_path='./train_10class/', input_shape=input_shape, class_num=class_num, transform=preprocess)
    test_ds = TinyImageNetDataset(src_path='./test_10class/', input_shape=input_shape, class_num=class_num, transform=preprocess)

    img, _ = train_ds[1]
    print(img.shape)
    print(len(train_ds))
    print(len(test_ds))

    train_meanRGB = [np.mean(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
    train_stdRGB = [np.std(x.numpy(), axis=(1, 2)) for x, _ in train_ds]
    train_meanR = np.mean([m[0] for m in train_meanRGB])
    train_meanG = np.mean([m[1] for m in train_meanRGB])
    train_meanB = np.mean([m[2] for m in train_meanRGB])
    train_stdR = np.mean([s[0] for s in train_stdRGB])
    train_stdG = np.mean([s[1] for s in train_stdRGB])
    train_stdB = np.mean([s[2] for s in train_stdRGB])

    print(train_meanR, train_meanG, train_meanB)
    print(train_stdR, train_stdG, train_stdB)
