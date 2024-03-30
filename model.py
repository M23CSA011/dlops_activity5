
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve, accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
from torch.utils.tensorboard import SummaryWriter

class CNNConfig1(nn.Module):
    def __init__(self):
        super(CNNConfig1, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256)  
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNConfig2(nn.Module):
    def __init__(self):
        super(CNNConfig2, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(16, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 4 * 4, 256) 
        self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv3(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class CNNConfig3(nn.Module):
    def __init__(self):
        super(CNNConfig3, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)  
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.relu(self.conv2(x))
        x = torch.max_pool2d(x, kernel_size=2, stride=2)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
transform = transforms.Compose([
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = torchvision.datasets.USPS(root='./content', train=True, transform=transform, download=True)
test_dataset = torchvision.datasets.USPS(root='./content', train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

batch = next(iter(train_loader))
batch[0].size()

batch_size=64

num_epochs=10
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

def compute_precision_recall(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    true_labels = label_binarize(true_labels, classes=range(10))
    predictions = label_binarize(predictions, classes=range(10))

    precision = dict()
    recall = dict()
    for i in range(10):
        precision[i], recall[i], _ = precision_recall_curve(true_labels[:, i], predictions[:, i])
    return precision, recall

def compute_metrics(true_labels, predictions):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, average='macro')
    recall = recall_score(true_labels, predictions, average='macro')
    cm = confusion_matrix(true_labels, predictions)
    return accuracy, precision, recall, cm

def train_model_cnn(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def evaluate_model_cnn(model, data_loader, device):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for images, labels in data_loader:
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    return predictions, true_labels

def train_and_evaluate_model(model, train_loader, test_loader, criterion, optimizer, device, num_epochs=10):
    for epoch in range(num_epochs):
        train_loss = train_model_cnn(model, train_loader, criterion, optimizer, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {train_loss}")

    predictions, true_labels = evaluate_model_cnn(model, test_loader, device)
    accuracy, precision, recall, cm = compute_metrics(true_labels, predictions)
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    precision_curve, recall_curve = compute_precision_recall(model, test_loader, device)

    writer = SummaryWriter("runs/precision_recall_curves")
    for i in range(10):
        writer.add_pr_curve(f"Model/Class_{i}", recall_curve[i], precision_curve[i], global_step=0)
    writer.close()

model_cnn1 = CNNConfig1().to(device)
criterion_cnn1 = nn.CrossEntropyLoss()
optimizer_cnn1 = optim.Adam(model_cnn1.parameters(), lr=0.001)
train_and_evaluate_model(model_cnn1, train_loader, test_loader, criterion_cnn1, optimizer_cnn1, device)

model_cnn2 = CNNConfig2().to(device)
criterion_cnn2 = nn.CrossEntropyLoss()
optimizer_cnn2 = optim.Adam(model_cnn2.parameters(), lr=0.001)
train_and_evaluate_model(model_cnn2, train_loader, test_loader, criterion_cnn2, optimizer_cnn2, device)

model_cnn3 = CNNConfig3().to(device)
criterion_cnn3 = nn.CrossEntropyLoss()
optimizer_cnn3 = optim.Adam(model_cnn3.parameters(), lr=0.001)
train_and_evaluate_model(model_cnn3, train_loader, test_loader, criterion_cnn3, optimizer_cnn3, device)
