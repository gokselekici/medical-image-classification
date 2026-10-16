import torch
import medmnist

print("PyTorch version:", torch.__version__)
print("MedMNIST installed successfully!")
print("CUDA available:", torch.cuda.is_available())

x = torch.rand(3, 3)
print("Random tensor:")
print(x)