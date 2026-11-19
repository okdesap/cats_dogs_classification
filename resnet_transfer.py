import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, Dataset, random_split

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(device)

#Data: the same split as the custom CNN 
root = "data/cats_and_dogs_kaggle"
dataset = datasets.ImageFolder(root=root)
train_raw, test_raw = random_split(dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42))

# ResNet was trained on ImageNet, so inputs must use the same normalization values
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    normalize,
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

#Model: pretrained ResNet18; only the final layer is trained
net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in net.parameters():
    param.requires_grad = False                # Freeze the backbone
net.fc = nn.Linear(net.fc.in_features, 2)      # New trainable output layer (512 -> 2)
net = net.to(device)

def set_train_mode(net):
    # Keep the backbone in eval mode to preserve ImageNet BatchNorm statistics
    net.train()
    for name, module in net.named_children():
        if name != "fc":
            module.eval()

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

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.fc.parameters(), lr=0.01, momentum=0.9)   # Only the final layer

epochs = 10
for epoch in range(epochs):
    set_train_mode(net)
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = loss_fn(net(inputs), labels)
        loss.backward()
        optimizer.step()

    tr = evaluate(train_loader, net, loss_fn)
    te = evaluate(test_loader, net, loss_fn)
    print(f"Epoch {epoch+1}: "
          f"Train Loss {tr[0]:.4f} Acc {tr[1]:.4f} | "
          f"Test Loss {te[0]:.4f} Acc {te[1]:.4f} Prec {te[2]:.4f} Rec {te[3]:.4f}")