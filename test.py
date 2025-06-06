# test.py

import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from dataset import LEVIRDataset
from model import remoteSensor
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define transforms
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Load test dataset
test_dataset = LEVIRDataset(root_dir='data/LEVIR-CD/test', transform=transform)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# Load model
model = remoteSensor().to(device)
model.load_state_dict(torch.load("checkpoint.pth", map_location=device))
model.eval()

# Display predictions for a few samples
import matplotlib.pyplot as plt

for idx, (img_A, img_B, mask) in enumerate(test_loader):
    if idx >= 5:  # Show only 5 samples
        break

    img_A, img_B, mask = img_A.to(device), img_B.to(device), mask.to(device)

    with torch.no_grad():
        output = model(img_A, img_B)
        pred = torch.sigmoid(output)
        pred = pred.squeeze().cpu().numpy()
        pred_binary = (pred > 0.5).astype(np.uint8)

    # Move mask to CPU and convert to numpy
    mask = mask.squeeze().cpu().numpy()

    # Plot
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    plt.title("Ground Truth")
    plt.imshow(mask, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Prediction")
    plt.imshow(pred_binary, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
