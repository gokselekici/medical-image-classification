from torch.utils.data import DataLoader
from medmnist import PneumoniaMNIST
from torchvision import transforms
from model import PneumoniaCNN
import torch.nn as nn
import torch.optim as optim

transform = transforms.ToTensor()

train_dataset = PneumoniaMNIST(
    split="train",
    transform=transform,
    download=True
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

print("Number of batches:", len(train_loader))

images, labels = next(iter(train_loader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)


model = PneumoniaCNN()
model.train()
criterion = nn.BCEWithLogitsLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    loss=0
    gesamtzahl = 0
    for images, y_ground in train_loader:
        optimizer.zero_grad()
        y_pred = model(images) # forward pass
        batch_loss = criterion(y_pred, y_ground.float())
        loss += (batch_loss.item()) * len(images)
        
        batch_loss.backward()
        gesamtzahl += len(images)
        
        optimizer.step()
    avg_loss = loss / gesamtzahl    
    print("average loss is: ", avg_loss)
        
    


