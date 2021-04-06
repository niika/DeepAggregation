from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torchvision
import numpy as np
from random import sample 
import re
import cv2

class ImageData(Dataset):
    def __init__(self, root, seq_length = 5, transform = None, distortion = None, size=(128,128)):
        self.root = root
        self.seq_length = seq_length
        
        all_imgs = os.listdir(root)
        self.image_paths = sorted(all_imgs)
        self.image_paths = [os.path.join(self.root, image_path) for image_path in self.image_paths] 
        self.images = [Image.open(img_loc).convert("RGB") for img_loc in self.image_paths]
        
        if transform is None:
            self.transform = transforms.Compose([transforms.RandomCrop(size),lambda x: np.array(x),transforms.ToTensor()])
        
        if distortion is None:
            self.distortion= transforms.Compose([transforms.ColorJitter(0.6,0.6,0.5,0.1),transforms.RandomErasing()])
        
    def __getitem__(self, index):
        y = self.images[index]
        if self.transform:
            y = self.transform(y)
        xs = [self.distortion(y.clone()) for _ in range(self.seq_length)]
        xs = torch.stack(xs)
        return xs, y
    
    def __len__(self):
        return len(self.images)
    
    def show(self, idx):
        return self.images[idx]
    
    
class SpecShadOcc(Dataset):
    def __init__(self, root, seq_length = 5, size=(256,256)):
        self.root = root    
        self.seq_length = seq_length
        
        batches = os.listdir(root)
        all_imgs_paths = [[root+"b{}/img{}.png".format(i,j)  for j in range(12) if os.path.isfile(root+"b{}/img{}.png".format(i,j))] for i in range(1,102) ]
        
        transform = transforms.Compose([transforms.Resize(size),lambda x: np.array(x),transforms.ToTensor()])
        self.images = list(map(lambda x: list(map( lambda elem: transform(Image.open(elem).convert("RGB")), x)), all_imgs_paths))
        
    def __getitem__(self, index):
        batch = self.images[index]
        y = batch[0]
        xs = sample(batch[1:],self.seq_length)
        xs = torch.stack(xs)
        return xs, y
    
    def getNitems(self, index, n):
        batch = self.images[index]
        y = batch[0]
        xs = sample(batch[1:],n)
        xs = torch.stack(xs)
        return xs, y
    
    def __len__(self):
        return len(self.images)
    
    def show(self, idx):
        toPIL = transforms.ToPILImage()
        return [toPIL(x) for x in self.images[idx]]
    
class SpecShadOccN(Dataset):
    def __init__(self, dataset, seq_length = 5):
        self.dataset = dataset
        self.seq_length = seq_length
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        return self.dataset.getNitems(index, self.seq_length)

    
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class CGData(Dataset):
    """Dataset containing all sequences with artifacts
    
    Generates three distorted images as input data.
    
    """
    
    def __init__(self, root_dir, indices, sample_size, resize=True):
        self.root_dir = root_dir
        self.ToPIL = transforms.ToPILImage()
        self.ToTensor = transforms.ToTensor()
        self.indices = indices
        self.resize=resize
        self.sample_size = sample_size
        
        files = os.listdir(root_dir)
        match = lambda x: len(re.findall("img_\d+_\d.jpg", x))== 1
        cut_string = lambda x: eval(re.sub("_.*","",re.sub("img_","",x)))

        files = list(filter(match,files))
        files = list(map(cut_string,files))


        first,last = min(files),max(files)
        self.offset = first
        self.last = last
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):        
        
        idx = self.indices[idx]
        count = 0
        img_files = None
        imgs = None
        label = None
        while True:
            try:
                n = self.sample_size
                nrs = np.random.choice(range(1,10), size=n, replace=False).tolist()
                img_files = [self.root_dir +  "img_" +str(idx)+ "_" + str(nr) + ".jpg" for nr in nrs]
                exists = all([os.path.isfile(img_file) for img_file in img_files])
                count+=1

                imgs = [cv2.imread(file) for file in img_files]
                imgs = [img[...,::-1]- np.zeros_like(img) for img in imgs]

                label_file = self.root_dir + "books/img " + "("+str(idx - 1)+").jpg"
                label = cv2.imread(label_file)
                label = label[...,::-1]- np.zeros_like(label)
                break

            except:
                idx = np.random.randint(len(self.indices))
                idx = self.indices[idx]

        
        
        if self.resize:
            label = cv2.resize(label, dsize=(256,256))
            imgs = [ cv2.resize(img, dsize=(256,256)) for img in imgs]
        
        H,W,C = imgs[0].shape
        if H<W:
            label = np.rot90(label)
            label -= np.zeros_like(label)
            imgs = [np.rot90(img) for img in imgs]- np.zeros_like(label)
        
        flip = np.random.randint(-1,3)
        if flip < 2:
            label = cv2.flip(label,flip)- np.zeros_like(label)
            imgs = [cv2.flip(img,flip) for img in imgs]- np.zeros_like(label)

    
        
        imgs = [transforms.ToTensor()(img) for img in imgs]
        imgs = torch.stack(imgs)
        
        label = label.astype(np.uint8)
        label = transforms.ToTensor()(label)
        #label = torch.unsqueeze(label,0)
        return imgs, label