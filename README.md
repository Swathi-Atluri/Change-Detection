
# 🛰️ Remote Sensing Change Detection Using Deep Learning

This project implements a deep learning pipeline for detecting structural and environmental changes between two temporally spaced remote sensing images using a U-Net-based semantic segmentation model. The pipeline is trained and evaluated on the LEVIR-CD dataset and can also be used to predict changes in any custom pair of input images.

---

## ✅ Overview

Change detection is crucial in applications such as urban development monitoring, disaster response, environmental protection, and defense. This project automates the process by using deep learning to compare two remote sensing images and predict pixel-level change masks.

The model has been trained using the LEVIR-CD dataset and is capable of generalizing to new image pairs. It supports GPU acceleration and is compatible with cloud-based platforms like Kaggle or local machines using Anaconda.

---

📥 Download the LEVIR-CD dataset from Kaggle: [Click here](https://www.kaggle.com/datasets/mdrifaturrahman33/levir-cd/data)

---

💡 **Tip:** Running this project on [Kaggle](https://www.kaggle.com/) can significantly speed up training since you get access to free GPU resources without any setup hassle.  
Note: You may need to modify file paths and environment-specific settings in the code to execute it properly on Kaggle.

---

## 💡 Technologies Used

| Tool / Library             | Purpose                                                           |
|----------------------------|-------------------------------------------------------------------|
| Python 3.8+                | Programming language                                              |
| PyTorch                    | Deep learning framework for building and training the model       |
| torchvision                | For image transformations and dataset handling                   |
| PIL / OpenCV               | Image loading and manipulation                                   |
| matplotlib                 | Visualization of predictions and results                         |
| Conda (Anaconda/Miniconda) | Virtual environment management                                   |

---

## 🗂️ Project Structure

```
📁 change-detection-project/
│
├── dataset.py            # Custom Dataset class for LEVIR-CD
├── model.py              # U-Net-based remoteSensor model architecture
├── train.py              # Training script
├── test.py               # Evaluates model on test data and shows predictions
├── custom_test.py        # Test any custom pair of images
├── requirements.txt      # Python dependencies
├── checkpoint.pth        # Trained model weights (after training)
└── README.md             # Project documentation
```

---

## ⚙️ Setup Instructions

Follow the steps below to run the project locally on your machine:

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/Swathi-Atluri/Change-Detection.git
```

### 2️⃣ Create and Activate a Virtual Environment

Using Miniconda or Anaconda:

Note: I already have it installed.

```bash
conda create -n cd_env python=3.8 -y
conda activate cd_env
```

### 3️⃣ Install Required Dependencies

```bash
pip install -r requirements.txt
```
---

## 🚀 How to Use

### 🏋️‍♀️ Train the Model

Place the LEVIR-CD dataset in the following format in your project folder:

```
data/LEVIR-CD/
│
├── train/
│   ├── A/
│   ├── B/
│   └── label/
├── val/
│   ├── A/
│   ├── B/
│   └── label/
├── test/
│   ├── A/
│   ├── B/
│   └── label/
```

Run the training script:

```bash
python train.py
```

The model checkpoint will be saved as `checkpoint.pth` in the working directory.

### 🧪 Test the Model on LEVIR-CD Test Set

```bash
python test.py
```

This will visualize a few test predictions alongside ground truth masks.

### 📸 Custom Image Inference

You can run the model on any custom image pair (outside of LEVIR-CD dataset).

Edit the file paths in `custom_test.py`:

```python
image_A_path = "./custom_data/before.jpg"
image_B_path = "./custom_data/after.jpg"
```

Run:

```bash
python custom_test.py
```

The script will automatically resize your images and generate a change mask prediction.

---

## 🧠 Model Architecture

The core model is a U-Net-based semantic segmentation network built from scratch and defined in `model.py` under the class name `remoteSensor`. It takes two RGB images as input and outputs a binary mask of changed regions using:

- Downsampling and upsampling layers
- Skip connections
- Sigmoid activation for final mask generation
- BCEWithLogitsLoss for training stability

---

## 📈 Outputs

- Visual difference maps between image A and B
- Ground truth vs predicted comparison
- Binary masks highlighting change
- Model saved as `checkpoint.pth`

---

