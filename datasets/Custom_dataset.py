from torch.utils.data import Dataset
from torch import tensor
from os import path,listdir
from numpy import zeros
from sklearn.model_selection import train_test_split
from PIL import Image

class Custom_dataset(Dataset):
     def __init__(self, mode = "train", root = "datasets/demo_dataset", transforms = None, test_size=0.2, val_size=0.1, random_state=42):
        super().__init__()
        self.mode = mode
        self.root = root
        self.transforms = transforms

        absolute_path = path.abspath(root)
        self.folder = absolute_path
        self.image_list = []
        self.label_list = []
        self.class_list = listdir(self.folder)
        self.class_list.sort()
        
        for class_id in range(len(self.class_list)):
            for image in listdir(path.join(self.folder, self.class_list[class_id])):
                self.image_list.append(path.join(self.folder, self.class_list[class_id], image))
                label = zeros(len(self.class_list))
                label[class_id] = 1.0
                self.label_list.append(label)

        train_imgs, test_imgs, train_labels, test_labels = train_test_split(
            self.image_list, self.label_list, test_size=test_size, random_state=42
        )
        train_imgs, val_imgs, train_labels, val_labels = train_test_split(
            train_imgs, train_labels, test_size=val_size / (1 - test_size), random_state=42
        )

        if self.mode == "train":
            self.image_list = train_imgs
            self.label_list = train_labels
        elif self.mode == "val":
            self.image_list = val_imgs
            self.label_list = val_labels
        elif self.mode == "test":
            self.image_list = test_imgs
            self.label_list = test_labels
        else:
            raise ValueError("Invalid mode value. Must be 'train', 'val', or 'test'.")
     def __getitem__(self, index):
        image_name = self.image_list[index]
        label = self.label_list[index]
        
        
        image = Image.open(image_name)
        if(self.transforms):
            image = self.transforms(image)
        
        label = tensor(label)
        
        return image, label
     def __len__(self):
        return len(self.image_list)  