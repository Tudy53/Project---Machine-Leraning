import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from sklearn.metrics import balanced_accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import torchvision
from PIL import Image
import os
import re
from sklearn.model_selection import KFold
from torch.utils.data import Subset, ConcatDataset
import copy
import PIL.Image

# 1. Load and process the data
transform = transforms.Compose([
    transforms.Resize((75, 75)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset_path = './TRAIN'
test_dataset_path = './TEST'

BATCH_SIZE = 32

# Remove all rotated images from the test dataset if they exist
for filename in os.listdir(test_dataset_path):
    # Check if the filename contains '_rotated'
    if '_rotated' in filename:
        # Construct the full file path
        file_path = os.path.join(test_dataset_path, filename)
        try:
            # Delete the file
            os.remove(file_path)
        except OSError as e:
            print(f"Error: {file_path} : {e.strerror}")

# Angles to rotate the images
angles = [5, -5, 10, -10, 20, -20]

# Adding the rotated images to the test dataset
if os.path.isdir(test_dataset_path):
    for filename in os.listdir(test_dataset_path):
        # Construct the full file path
        file_path = os.path.join(test_dataset_path, filename)

        # Check if it's a file
        if os.path.isfile(file_path):
            try:
                with Image.open(file_path) as img:
                    # Rotate and save each image
                    for angle in angles:
                        rotated_img = img.rotate(angle, expand=True)
                        new_filename = f"{os.path.splitext(filename)[0]}_rotated{angle}{os.path.splitext(filename)[1]}"
                        new_file_path = os.path.join(test_dataset_path, new_filename)
                        rotated_img.save(new_file_path)
            except IOError:
                print(f"Could not process file {file_path} as an image.")

                
train_datagen = datasets.ImageFolder(root=train_dataset_path, transform=transform)
train_dataset_loader = DataLoader(train_datagen, batch_size=BATCH_SIZE, shuffle=True) 

# Mapping numeric labels to class names
the_real_labels = {}
with open("./labels.csv", "r") as label_f:
    for line in label_f.readlines()[1:]:
        label_value, label_description = line.strip().split(";")
        the_real_labels[int(label_value)] = label_description

# 2. Define the Neural Network Architecture

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=9):
        super(SimpleCNN, self).__init__()
        # conv1: The first convolutional layer (nn.Conv2d) takes 3 input channels (assuming RGB images), outputs 64 channels, and uses a 3x3 kernel with stride 1 and padding 1. Padding of 1 maintains the spatial dimensions of the input assuming the stride is 1.
        # pool: A max pooling layer (nn.MaxPool2d) that reduces the spatial dimensions by half (kernel size and stride are both 2).
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        # conv2: The second convolutional layer reduces the number of channels from 64 to 32, maintaining the spatial dimensions due to the same kernel size, stride, and padding as the first convolutional layer.
        self.conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        # conv3: A third convolutional layer that further reduces the channels from 32 to 16, again maintaining spatial dimensions.
        self.conv3 = nn.Conv2d(32, 16, kernel_size=3, stride=1, padding=1)
        # fc1: The first fully connected layer (nn.Linear) maps the flattened features from the pooled output of conv2 to 128 features. It seems there's a mismatch in the layer connection here as it should connect the output from conv3 instead.
        self.fc1 = nn.Linear(32*18*18, 128)
        # dropout: A dropout layer with a dropout probability of 0.3, which helps prevent overfitting by randomly setting input elements to zero during training.
        self.dropout = nn.Dropout(0.3)
        # fc2 and fc3: Further fully connected layers to reduce the dimension from 128 to 64 and finally to num_classes, which is the number of classes for the output layer.
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        
    def forward(self, x):
        # Apply the first convolution, followed by pooling
        x = self.pool(nn.functional.relu(self.conv1(x)))
        # Apply the second convolution, followed by pooling
        x = self.pool(nn.functional.relu(self.conv2(x)))
        # Flatten the output for the fully connected layer
        x = x.view(-1, 32 * 18 * 18)
        # Apply the first fully connected layer
        x = nn.functional.relu(self.fc1(x))
        # Output layer
        x = self.fc2(x)
        return x
    

def train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, patience):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    epochs_no_improve = 0

    for epoch in range(n_epochs):
        # Training phase
        model.train()
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

        # Validation phase
        model.eval()
        all_preds, all_targets = [], []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.tolist())
                all_targets.extend(y.tolist())

        val_acc = balanced_accuracy_score(all_targets, all_preds)
        print(f'Epoch {epoch+1}/{n_epochs}, Val Accuracy: {val_acc:.4f}')

        # Early stopping and learning rate adjustment
        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        scheduler.step()  # Adjust the learning rate based on the scheduler

        if epochs_no_improve == patience:
            print('Early stopping')
            break

    model.load_state_dict(best_model_wts)
    return model


def k_fold_training(model, dataset, k=5, n_epochs=100, patience=10, optimizer=None, scheduler=None):
    kf = KFold(n_splits=k, shuffle=True)
    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'Fold {fold+1}/{k}')
        train_subs = Subset(dataset, train_idx)
        val_subs = Subset(dataset, val_idx)
        train_loader = DataLoader(train_subs, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_subs, batch_size=BATCH_SIZE)

        model = train_fold(model, train_loader, val_loader, criterion, optimizer, scheduler, n_epochs, patience)

# Prepare the complete dataset for K-Fold CV
complete_dataset = ConcatDataset([train_datagen])
my_model = SimpleCNN(9)

criterion = nn.CrossEntropyLoss()

optimizer_model = optim.Adam(my_model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer_model, step_size=10, gamma=0.1)

# Run K-Fold Training
k_fold_training(my_model, complete_dataset, k=5, n_epochs=100, patience=10, optimizer=optimizer_model, scheduler=scheduler)

class UnlabeledDataset(Dataset):
    def __init__(self, images_folder, transform=None):
        self.images_folder = images_folder
        self.image_files = [os.path.join(images_folder, f) for f in os.listdir(images_folder) if os.path.isfile(os.path.join(images_folder, f))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# Load the test dataset
test_dataset = UnlabeledDataset(images_folder=test_dataset_path, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to make predictions
# Predict function modified to use class names
def predict(model, data_loader, device):
    model.eval()  # Set the model to evaluation mode
    predictions = []
    with torch.no_grad():
        for images in data_loader:  # Corrected to expect only one value
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            # Convert numeric predictions to actual class names using 'the_real_labels'
            for pred in predicted:
                if pred == 0:
                    predictions.append("12")
                elif pred == 1:
                    predictions.append("13")
                elif pred == 2:
                    predictions.append("24")
                elif pred == 3:
                    predictions.append("37")
                elif pred == 4:
                    predictions.append("38")
                elif pred == 5:
                    predictions.append("39")
                elif pred == 6:
                    predictions.append("44")
                elif pred == 7:
                    predictions.append("50")
                elif pred == 8:
                    predictions.append("6")
    return predictions

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
my_model.to(device)

model_predictions = predict(my_model, test_loader, device)

names = [f for f in os.listdir(test_dataset_path) if os.path.isfile(os.path.join(test_dataset_path, f))]

from collections import Counter

train_labels = [label for _, label in train_datagen]
label_counts = Counter(train_labels)

def save_predictions(predictions, filename):
    with open(filename, "w") as f:
        f.write("ID,TARGET\n")
        for i, pred in enumerate(predictions):
            if pred == "Unknown":
                f.write(f"{names[i].split('.')[0]} Unknown prediction for image {names[i].split('.')[0]}\n")
            else:
                # Correctly write the prediction to the file
                f.write(f"{names[i].split('.')[0]},{pred}\n")

save_predictions(model_predictions, "predictions.csv")

word_to_remove = "rotated"
input_file_path = 'predictions.csv'
output_file_path = 'predictions1.csv'

with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    for line in input_file:
        if word_to_remove not in line:
            output_file.write(line)

# Replace the original file with the new file
os.replace(output_file_path, input_file_path)
