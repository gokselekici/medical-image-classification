import medmnist
from medmnist import PneumoniaMNIST
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.ToTensor()

train_dataset = PneumoniaMNIST(
    split="train",
    transform=transform,
    download=True
)

#Gib aus, wie viele Bilder der Trainingsdatensatz enthält.
print(len(train_dataset))
#Untersuche die Form (shape) eines einzelnen Bildes und seines Labels.
image, label = train_dataset[0]
print(image.shape)
print(label)
#Zeige mit matplotlib die ersten sechs Bilder nebeneinander an, jeweils mit dem zugehörigen Label.

fig, axes = plt.subplots(1, 6, figsize=(12, 3))

for i in range(6):
    image, label = train_dataset[i]

    image = np.squeeze(image)
    # TODO 2: Zeige das Bild auf axes[i] an.
    axes[i].imshow(image, cmap="gray")
    # TODO 3: Setze das Label als Titel.
    axes[i].set_title(f"Label: {label.item()}")
    axes[i].axis("off")

plt.tight_layout()
plt.show()

cnt = 0
for i in range(len(train_dataset)):
    label = train_dataset[i][1]
    if label == 1:
        cnt = cnt+1

pneumonia_count = cnt
normal_count = len(train_dataset) - cnt
print(pneumonia_count, " images are with pneummonia")  
print(normal_count, " images are healthy")      
print(f"{(pneumonia_count / len(train_dataset) * 100):.2f}% of the images are with pneumonia")

classes = ["Healthy", "Pneumonia"]
counts = [normal_count, pneumonia_count]

plt.bar(classes, counts)

# TODO 1: Titel hinzufügen
plt.title("Class Distribution in PneumoniaMNIST Training Set")
# TODO 2: Y-Achse beschriften
plt.ylabel("Number of Images")
# TODO 3: Diagramm anzeigen
plt.show()


val_dataset = PneumoniaMNIST(
    split="val",
    transform=transform,
    download=True
)
test_dataset = PneumoniaMNIST(
    split="test",
    transform=transform,
    download=True
)


def count_classes(dataset):
    normal_count = 0
    pneumonia_count = 0

    for i in range(len(dataset)):
        label = dataset[i][1]
        if label == 1:
            pneumonia_count = pneumonia_count + 1
        else:
            normal_count = normal_count + 1

    return normal_count, pneumonia_count

print(count_classes(test_dataset))
print(count_classes(val_dataset))
print(count_classes(train_dataset))

