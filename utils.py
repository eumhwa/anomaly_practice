from PIL import Image
import torchvision.transforms as transforms

from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset, DataLoader


def get_standard_transform():
    trans = transforms.Compose(
        [
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return trans

def get_transform(img_size):
    trans = transforms.Compose(
        [
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    return trans
    
def load_MVTecAD_dataset(dataset_path, is_train, batch_size=1, transform=None):
    if is_train:
        dataset_path = os.path.join(dataset_path, "train")
    else:
        dataset_path = os.path.join(dataset_path, "test")

    if transform is None:
        target_transform = get_standard_transform()
    else:
        target_transform = transform

    dataset = ImageFolder(root=dataset_path, transform=target_transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)        

class CustomDataset(Dataset):
    # Custom Dataset class    
    def __init(self, file_list, labels, transform=None):
        self.file_list = file_list
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image = Image.open(self.file_list[idx])
        label = self.labels[idx]
        
        if self.transform is None:
            self.transform = get_standard_transform()
            image_transform = self.transform(image)
        
        return image_transform, label
