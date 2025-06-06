import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model import remoteSensor  # Your model class
import sys

# ========== CONFIGURATION ==========
# Provide paths to your custom images (before and after)
image_A_path = ".\data\LEVIR-CD\test\A\test_11.png"  # e.g., "samples/before.png"
image_B_path = ".\data\LEVIR-CD\test\B\test_11.png"  # e.g., "samples/after.png"
model_path = "model.pth"  # Path to the trained model weights
image_size = (256, 256)  # Must match training size
# ====================================

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
])

# Load and transform images
def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)  # Add batch dimension

img_A = load_image(image_A_path)
img_B = load_image(image_B_path)

# Load model
model = remoteSensor().to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Predict
with torch.no_grad():
    output = model(img_A, img_B)
    output = torch.sigmoid(output)
    pred_mask = output.squeeze().cpu().numpy()

# Load original images for display
original_A = Image.open(image_A_path).resize(image_size)
original_B = Image.open(image_B_path).resize(image_size)

# Plot results
plt.figure(figsize=(12, 4))
plt.subplot(1, 3, 1)
plt.imshow(original_A)
plt.title("Before (A)")
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(original_B)
plt.title("After (B)")
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(pred_mask, cmap='gray')
plt.title("Predicted Change Mask")
plt.axis('off')

plt.tight_layout()
plt.show()
