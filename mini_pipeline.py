import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from show_masks import OxfordPetsDataset  # dein Dataset

# ------------------------
# 1. Dataset + DataLoader
# ------------------------
transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor()
])

dataset = OxfordPetsDataset(root="./oxford_pets", transform=transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

# ------------------------
# 2. Mini-CNN Modell
# ------------------------
class SimpleSegmentationModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, num_classes, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x

model = SimpleSegmentationModel(num_classes=3)

# ------------------------
# 3. Loss + Optimizer
# ------------------------
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# ------------------------
# 4. Trainingsschleife
# ------------------------
num_epochs = 10

for epoch in range(num_epochs):
    running_loss = 0.0
    for i, (images, masks) in enumerate(loader):
        masks = (masks.squeeze(1) - 1).long()  # Labels 0,1,2
        
        outputs = model(images)
        loss = criterion(outputs, masks)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}")
    
    # ------------------------
    # 5. Predicted Masks visualisieren
    # ------------------------
    with torch.no_grad():
        sample_images, sample_masks = next(iter(loader))
        sample_masks = (sample_masks.squeeze(1) - 1).long()
        sample_outputs = model(sample_images)
        pred_masks = torch.argmax(sample_outputs, dim=1)

        # erstes Bild visualisieren
        img = sample_images[0].permute(1,2,0).numpy()
        true_mask = sample_masks[0].numpy()
        pred_mask = pred_masks[0].numpy()

        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.imshow(img)
        plt.title("Image")
        plt.subplot(1,3,2)
        plt.imshow(true_mask, cmap='gray')
        plt.title("Ground Truth Mask")
        plt.subplot(1,3,3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted Mask")
        plt.show()
