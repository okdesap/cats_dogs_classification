import torch
from torch import nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset, random_split

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

root = "data/cats_and_dogs_kaggle"
dataset = datasets.ImageFolder(root=root)
train_raw, test_raw = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),   # Augmentation 1
    transforms.RandomRotation(10),       # Augmentation 2
    transforms.ToTensor(),
])
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class WithTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
    def __len__(self):
        return len(self.subset)
    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

train_loader = DataLoader(WithTransform(train_raw, train_transform), batch_size=32, shuffle=True)
test_loader = DataLoader(WithTransform(test_raw, test_transform), batch_size=32)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.conv3 = nn.Conv2d(32, 64, 3)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.flatten = nn.Flatten()
        self.fcl = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 2)
    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.flatten(x)
        x = self.relu(self.fcl(x))
        return self.fc2(x)

def evaluate(loader, net, loss_fn):
    net.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            total_loss += loss_fn(outputs, labels).item()
            predicted = outputs.argmax(1)
            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()
    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    return total_loss / len(loader), accuracy, precision, recall

net = NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)

epochs = 30
for epoch in range(epochs):
    net.train()
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(net(inputs), labels)
        loss.backward()
        optimizer.step()

    tr_loss, tr_acc, tr_prec, tr_rec = evaluate(train_loader, net, loss_fn)
    te_loss, te_acc, te_prec, te_rec = evaluate(test_loader, net, loss_fn)
    print(f"Epoch {epoch+1}: "
          f"Train Loss {tr_loss:.4f} Acc {tr_acc:.4f} Prec {tr_prec:.4f} Rec {tr_rec:.4f} | "
          f"Test Loss {te_loss:.4f} Acc {te_acc:.4f} Prec {te_prec:.4f} Rec {te_rec:.4f}")
    torch.save(net.state_dict(), f"checkpoint_epoch_{epoch+1}.pth")

# Load the checkpoint and verify that it works.
best = NeuralNetwork().to(device)
best.load_state_dict(torch.load(f"checkpoint_epoch_{epochs}.pth", weights_only=True, map_location=device))
te_loss, te_acc, te_prec, te_rec = evaluate(test_loader, best, loss_fn)
print(f"Loaded checkpoint (epoch {epochs}): "
      f"Test Loss {te_loss:.4f} Acc {te_acc:.4f} Prec {te_prec:.4f} Rec {te_rec:.4f}")