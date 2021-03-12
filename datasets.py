from PIL import Image
import torch
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import os
from torchvision import transforms
import torchvision
import numpy as np
from random import sample 

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
    
    def __len__(self):
        return len(self.images)
    
    def show(self, idx):
        toPIL = transforms.ToPILImage()
        return [toPIL(x) for x in self.images[idx]]