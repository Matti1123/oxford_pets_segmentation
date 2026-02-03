import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt

class OxfordPetsDataset(Dataset):
    def __init__(self, root, transform=None):
        self.img_dir = os.path.join(root, "images")
        self.mask_dir = os.path.join(root, "annotations", "trimaps")
        self.transform = transform
        # Nur echte JPGs laden
        self.images = sorted([f for f in os.listdir(self.img_dir) if f.endswith(".jpg")])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace(".jpg", ".png"))
        
        # Bild laden
        image = Image.open(img_path).convert("RGB")
        # Maske laden
        mask = Image.open(mask_path)

        if self.transform:
            # Bild resize + Tensor
            image = self.transform(image)
            # Maske ebenfalls resize (NEAREST = keine Interpolation der Klassen)
            mask = mask.resize((128,128), resample=Image.NEAREST)
            mask = np.array(mask)
            mask = torch.tensor(mask, dtype=torch.long)
        
        return image, mask


# Transformation: resize + normalize
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = OxfordPetsDataset(root="./oxford_pets", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test: erstes Bild + Maske anzeigen
images, masks = next(iter(loader))
img = images[0].permute(1,2,0)  # Channels last
mask = masks[0]                  # Maske als 2D Tensor

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")
plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask")
plt.show()
