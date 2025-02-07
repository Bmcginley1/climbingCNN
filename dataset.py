import numpy as np
import torch
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import os
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CustomImageDataset(Dataset):
    def __init__(self, csv_file = "climbing_data_use_this_one.csv", image_dir = "enter_file_destination", transform=None): # use own file destination
        self.annotations = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
    def __len__(self):
        return len(self.annotations)
    def __getitem__(self, index):
        img_name = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image = Image.open(img_name).convert('RGB')
        label = int(self.annotations.iloc[index, 1])
        if self.transform:
            image = self.transform(image)
        return image, label

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()  # Normalize
])
csv_file = "climbing_data_use_this_one.csv"
image_dir = "C:/Users/14077/Documents/Climbing Hold Images"
dataset = CustomImageDataset(csv_file = csv_file, image_dir = image_dir, transform = transform)




train_size = int(0.85 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])




train_loader = torch.utils.data.DataLoader(train_dataset, batch_size = 32, shuffle = True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 32, shuffle = False)

for i, (train_images, train_labels) in enumerate(train_loader):
    train_images, train_labels = train_images.to(device), train_labels.to(device)
    print(f"Batch {i+1}")
    print(f"Images batch shape: {train_images.size()}")
    print(f"Labels batch: {train_labels}")
    if i == 0:
        image = train_images[0].cpu().permute(1, 2, 0).numpy()
        plt.imshow(image)
        plt.title(f"Label: {train_labels[0]}")
        plt.show()
    if i == 1:
        break


class MyConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_p):
        kernel_size = 5
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
    def forward(self, x):
        return self.model(x)




class Model(nn.Module):
    def __init__(self, n_classes=6):
        super().__init__()
        self.image_branch = nn.Sequential(
            MyConvBlock(3, 32, 0.2),
            MyConvBlock(32, 64, 0.2),
            MyConvBlock(64, 128, 0.2),
        )
        self.dummy_input = torch.zeros(1, 3, 128, 128)
        self.flattened_size = self._get_flattened_size(self.dummy_input)

        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, n_classes)
        self.to(device)
    def _get_flattened_size(self, input_tensor):
        x = self.image_branch(input_tensor)
        return x.view(x.size(0), -1).size(1)

    def forward(self, image):
        image_features = self.image_branch(image)
        x = torch.flatten(image_features, 1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

model = Model(n_classes = 6).to(device)


loss_function = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00005, weight_decay=1e-3)
scheduler = ReduceLROnPlateau(optimizer, mode='max', patience=5, factor=0.5)
def get_batch_accuracy(output, y, N):
    pred = output.argmax(dim=1, keepdim=True)
    correct = pred.eq(y.view_as(pred)).sum().item()
    return correct / N

def train():
    loss = 0
    accuracy = 0
    total = 0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)  # Move inputs and labels to the same device as the model
        output = model(x)
        optimizer.zero_grad()
        batch_loss = loss_function(output, y)
        batch_loss.backward()
        optimizer.step()
        loss += batch_loss.item() * x.size(0)
        accuracy += get_batch_accuracy(output, y, x.size(0)) * x.size(0)
        total += x.size(0)
    avg_loss = loss / total
    avg_accuracy = accuracy / total
    print(f'Train - Loss: {avg_loss:.4f} Accuracy: {avg_accuracy:.4f}')
    return avg_loss, avg_accuracy

def test():
    loss = 0
    accuracy = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)  # Move inputs and labels to the same device as the model
            output = model(x)
            batch_loss = loss_function(output, y)
            loss += batch_loss.item() * x.size(0)
            accuracy += get_batch_accuracy(output, y, x.size(0)) * x.size(0)
            total += x.size(0)
        avg_loss = loss / total
        avg_accuracy = accuracy / total
    print(f'Valid - Loss: {avg_loss:.4f} Accuracy: {avg_accuracy:.4f}')
    return avg_loss, avg_accuracy

def calculate_accuracy(model, dataloader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, predicted = torch.max(output, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    overall_accuracy = correct / total
    return overall_accuracy


epochs = 32
best_val_loss = np.inf
epochs_no_improve = 0
patience = 5
delta = .001
best_model_wts = None

for epoch in range(epochs):
    print(f"Epoch {epoch + 1}/{epochs}")


    train_loss, train_accuracy = train()
    val_loss, val_accuracy = test()

    if best_val_loss - val_loss > delta:
        best_val_loss = val_loss
        epochs_no_improve = 0
        best_model_wts = model.state_dict()
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print("Early stopping triggered!")
        model.load_state_dict(best_model_wts)  # Restore the best weights
        break

test_accuracy = calculate_accuracy(model, test_loader, device)
print(f"Overall Test Accuracy: {test_accuracy}")
