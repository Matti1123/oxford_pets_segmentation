import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from show_masks import OxfordPetsDataset  # deine Dataset-Klasse

# Transformation: Resize + Tensor
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

# Dataset
dataset = OxfordPetsDataset(root="./oxford_pets", transform=transform)

# DataLoader
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# Test: ersten Batch laden
images, masks = next(iter(loader))

print(f"Images shape: {images.shape}")  # [B, C, H, W]
print(f"Masks shape:  {masks.shape}")   # [B, 1, H, W]

# Erstes Sample visualisieren
img = images[0].permute(1,2,0)  # Channels-last
mask = masks[0].squeeze(0)             # erster Kanal

plt.figure(figsize=(8,4))
plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Image")
plt.subplot(1,2,2)
plt.imshow(mask)
plt.title("Mask")
plt.show()
