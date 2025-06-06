import torch

if torch.cuda.is_available():
    print("CUDA is available!")
    print("Device name:", torch.cuda.get_device_name(0))
else:
    print("CUDA NOT available. Running on CPU.")