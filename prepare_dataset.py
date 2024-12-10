import requests
import os
from io import BytesIO
from zipfile import ZipFile
from torch.utils.data import Dataset, DataLoader, random_split, TensorDataset
from torchvision import transforms, utils
import pathlib
from PIL import Image

def extract_data():
    # the dataset includes the images as standard 3-band RGB (red, green, blue) images
    rgb_data_url = "https://zenodo.org/record/7711810/files/EuroSAT_RGB.zip"

    response_rgb = requests.get(rgb_data_url)
    # response_multiband = requests.get(multiband_data_url)

    # Raise any exceptions if the http requests rendered any errors
    #response_rbg.raise_for_status()


    # Extract the zip files
    # os.makedirs('Multi', exist_ok=False)
    os.makedirs('RGB', exist_ok=False)
    '''with ZipFile(BytesIO(response_multiband.content)) as zip_file:
    zip_file.extractall(f"{path}RBG")'''

    with ZipFile(BytesIO(response_rgb.content)) as zip_file:
        zip_file.extractall(f"RGB")

def prepare_data(root_dir):
    image_paths = []
    labels = []
    for _, class_name in enumerate(sorted(os.listdir(root_dir))):
        class_dir = os.path.join(root_dir, class_name)
        for file_name in os.listdir(class_dir):
            if file_name.endswith(".jpg") or file_name.endswith('.tif'):
                image_paths.append(os.path.join(class_dir, file_name))
                labels.append(class_name)
    return image_paths, labels

class EuroSATDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.label_dict = {"AnnualCrop": 0, "Forest":1, "HerbaceousVegetation": 2,
                    "Highway": 3, "Industrial": 4, "Pasture": 5, "PermanentCrop": 6,
                    "Residential": 7, "River": 8, "SeaLake": 9}
        self.labels = [self.label_dict[label] for label in labels]
        self.transform = transform

    def _load_image(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = self._load_image(idx)
        label = self.labels[idx]
        return image, label

def load_dataset(root_dir):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # load in the dataset
    image_paths, labels = prepare_data(root_dir=root_dir)

    # --- split the data between train, valid, and test ---
    X_train, X_test, y_train, y_test = train_test_split(image_paths, labels,
                                                        test_size=0.2, random_state=84, stratify = labels)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test,
                                                        test_size=0.5, random_state=84, stratify = y_test)


    train_dataset = EuroSATDataset(image_paths=X_train, labels=y_train, transform=transform)
    valid_dataset = EuroSATDataset(image_paths=X_valid, labels=y_valid, transform=transform)
    test_dataset = EuroSATDataset(image_paths=X_test, labels=y_test, transform=transform)
    return  image_paths, labels, train_dataset, valid_dataset, test_dataset

def load_dataloaders(test_dataset, train_dataset, valid_dataset):
    test_load = DataLoader(test_dataset, batch_size = 32, shuffle = False, num_workers=os.cpu_count())
    train_load = DataLoader(train_dataset, batch_size = 32, shuffle = True, num_workers=os.cpu_count())
    valid_load = DataLoader(valid_dataset, batch_size = 32, shuffle = False, num_workers=os.cpu_count())
    return test_load, train_load, valid_load
