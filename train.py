# train.py  (with live progress bar)
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm                                   # ← progress bar
from dataset import LEVIRDataset
from model import remoteSensor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# datasets & loaders
train_dataset = LEVIRDataset(root_dir='data/LEVIR-CD/train', transform=transform)
val_dataset   = LEVIRDataset(root_dir='data/LEVIR-CD/val',   transform=transform)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=8, shuffle=False)

# model / loss / opt
model = remoteSensor().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

num_epochs = 20
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    # tqdm wraps the loader to show a live progress bar
    for img_A, img_B, mask in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
        img_A, img_B, mask = img_A.to(device), img_B.to(device), mask.to(device)

        optimizer.zero_grad()
        outputs = model(img_A, img_B)
        loss = criterion(outputs, mask)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * img_A.size(0)
        tqdm.write(f"batch loss: {loss.item():.4f}")      # live batch loss (optional)

    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} ▸ Train loss: {epoch_loss:.4f}")

    # ---------- validation ----------
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for img_A, img_B, mask in val_loader:
            img_A, img_B, mask = img_A.to(device), img_B.to(device), mask.to(device)
            outputs = model(img_A, img_B)
            loss = criterion(outputs, mask)
            val_loss += loss.item() * img_A.size(0)

    avg_val_loss = val_loss / len(val_loader.dataset)
    print(f"Epoch {epoch+1}/{num_epochs} ▸ Val loss:   {avg_val_loss:.4f}\n")

print("Training complete!")
