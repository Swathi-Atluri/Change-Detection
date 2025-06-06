# Import necessary libraries
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T

class LEVIRDataset(Dataset):
    
    def __init__(self, root_dir, mode='train', transform=None):
        self.image_dir_A = os.path.join(root_dir, 'A')
        self.image_dir_B = os.path.join(root_dir, 'B')
        self.mask_dir = os.path.join(root_dir, 'label')

        self.transform = transform
        self.filenames = sorted(os.listdir(self.image_dir_A))
        
    def __len__(self):
        return len(self.filenames)
    
    def __getitem__(self, idx):
        img_name = self.filenames[idx]

        path_A = os.path.join(self.image_dir_A, img_name)
        path_B = os.path.join(self.image_dir_B, img_name)
        path_mask = os.path.join(self.mask_dir, img_name)

        image_A = Image.open(path_A).convert('RGB')
        image_B = Image.open(path_B).convert('RGB')
        mask = Image.open(path_mask).convert('L')  # grayscale

        if self.transform:
            image_A = self.transform(image_A)
            image_B = self.transform(image_B)

            mask = mask.resize((256, 256), Image.NEAREST)  # resize mask to match input
            mask = T.ToTensor()(mask)


        return image_A, image_B, mask
