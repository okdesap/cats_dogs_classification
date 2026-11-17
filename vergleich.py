"""
Vergleich: eigenes CNN (von Grund auf) vs. ResNet18 (Transfer Learning)
auf dem Cats-and-Dogs-Datensatz.

Das Skript trainiert beide Modelle mit demselben Split (Seed 42) und speichert in results/:
  - curves.png          Loss- und Accuracy-Kurven (Train vs. Test)
  - confusion.png       Confusion Matrix beider Modelle
  - misclassified.png   falsch klassifizierte Testbilder des ResNet18
  - results.md          Ergebnistabelle (Accuracy, Precision, Recall, F1, Parameter, Zeit)

Start (aus dem Ordner, in dem data/cats_and_dogs liegt):
    python portfolio_results.py
"""
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print("Device:", device)

ROOT = "data/cats_and_dogs"
OUT = "results"
os.makedirs(OUT, exist_ok=True)
CLASSES = ["cat", "dog"]  # ImageFolder sortiert alphabetisch: cats=0, dogs=1

# ---------------------------------------------------------------- Daten
dataset = datasets.ImageFolder(root=ROOT)
train_raw, test_raw = random_split(
    dataset, [0.8, 0.2], generator=torch.Generator().manual_seed(42)
)


class MitTransform(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label


def make_loaders(use_imagenet_norm):
    extra = [transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])] if use_imagenet_norm else []
    train_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        *extra,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        *extra,
    ])
    train_loader = DataLoader(MitTransform(train_raw, train_tf), batch_size=32, shuffle=True)
    test_loader = DataLoader(MitTransform(test_raw, test_tf), batch_size=32, shuffle=False)
    return train_loader, test_loader


# ---------------------------------------------------------------- Modelle
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


def build_resnet():
    net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for p in net.parameters():
        p.requires_grad = False
    net.fc = nn.Linear(net.fc.in_features, 2)
    return net


def train_mode_cnn(net):
    net.train()


def train_mode_resnet(net):
    # Backbone im eval-Modus lassen, damit die BatchNorm-Statistiken von ImageNet erhalten bleiben
    net.train()
    for name, module in net.named_children():
        if name != "fc":
            module.eval()


# ---------------------------------------------------------------- Auswertung (Metriken manuell)
def evaluate(loader, net, loss_fn, return_preds=False):
    net.eval()
    tp = fp = tn = fn = 0
    total_loss = 0.0
    all_pred, all_true = [], []
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
            if return_preds:
                all_pred += predicted.cpu().tolist()
                all_true += labels.cpu().tolist()
    n = tp + fp + tn + fn
    acc = (tp + tn) / n if n > 0 else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    result = {
        "loss": total_loss / len(loader), "acc": acc, "prec": prec, "rec": rec, "f1": f1,
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }
    if return_preds:
        result["pred"], result["true"] = all_pred, all_true
    return result


def run(name, net, optimizer, train_mode, train_loader, test_loader, epochs):
    net = net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    hist = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    start = time.time()
    for epoch in range(epochs):
        train_mode(net)
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = loss_fn(net(inputs), labels)
            loss.backward()
            optimizer.step()
        tr = evaluate(train_loader, net, loss_fn)
        te = evaluate(test_loader, net, loss_fn)
        hist["train_loss"].append(tr["loss"])
        hist["train_acc"].append(tr["acc"])
        hist["test_loss"].append(te["loss"])
        hist["test_acc"].append(te["acc"])
        print(f"[{name}] Epoche {epoch + 1}/{epochs}: "
              f"Train Loss {tr['loss']:.4f} Acc {tr['acc']:.4f} | "
              f"Test Loss {te['loss']:.4f} Acc {te['acc']:.4f}")
    seconds = time.time() - start
    final = evaluate(test_loader, net, loss_fn, return_preds=True)
    params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {"hist": hist, "final": final, "seconds": seconds, "params": params, "net": net}


# ---------------------------------------------------------------- Training beider Modelle
cnn_train, cnn_test = make_loaders(use_imagenet_norm=False)
cnn = NeuralNetwork()
cnn_res = run("CNN", cnn, torch.optim.SGD(cnn.parameters(), lr=0.005, momentum=0.9),
              train_mode_cnn, cnn_train, cnn_test, epochs=30)

rn_train, rn_test = make_loaders(use_imagenet_norm=True)
resnet = build_resnet()
rn_res = run("ResNet18", resnet, torch.optim.SGD(resnet.fc.parameters(), lr=0.01, momentum=0.9),
             train_mode_resnet, rn_train, rn_test, epochs=10)

results = {"Eigenes CNN": cnn_res, "ResNet18 (Transfer)": rn_res}

# ---------------------------------------------------------------- Plot 1: Kurven
fig, axes = plt.subplots(2, 2, figsize=(11, 7))
for row, (name, r) in enumerate(results.items()):
    h = r["hist"]
    ep = range(1, len(h["train_loss"]) + 1)
    axes[row, 0].plot(ep, h["train_loss"], label="Train")
    axes[row, 0].plot(ep, h["test_loss"], label="Test")
    axes[row, 0].set_title(f"{name}: Loss")
    axes[row, 1].plot(ep, h["train_acc"], label="Train")
    axes[row, 1].plot(ep, h["test_acc"], label="Test")
    axes[row, 1].set_title(f"{name}: Accuracy")
    for ax in axes[row]:
        ax.set_xlabel("Epoche")
        ax.grid(alpha=0.3)
        ax.legend()
fig.tight_layout()
fig.savefig(f"{OUT}/curves.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- Plot 2: Confusion Matrix
fig, axes = plt.subplots(1, 2, figsize=(9, 4))
for ax, (name, r) in zip(axes, results.items()):
    f = r["final"]
    matrix = [[f["tn"], f["fp"]], [f["fn"], f["tp"]]]  # Zeilen: wahr, Spalten: vorhergesagt
    ax.imshow(matrix, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, matrix[i][j], ha="center", va="center", fontsize=14,
                    color="white" if matrix[i][j] > max(map(max, matrix)) / 2 else "black")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(CLASSES)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("vorhergesagt")
    ax.set_ylabel("wahr")
    ax.set_title(name)
fig.tight_layout()
fig.savefig(f"{OUT}/confusion.png", dpi=150)
plt.close(fig)

# ---------------------------------------------------------------- Plot 3: Fehlklassifikationen (ResNet18)
pred, true = rn_res["final"]["pred"], rn_res["final"]["true"]
wrong = [i for i in range(len(pred)) if pred[i] != true[i]][:8]
if wrong:
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for ax in axes.flat:
        ax.axis("off")
    for ax, i in zip(axes.flat, wrong):
        img, _ = test_raw[i]  # Originalbild (PIL); Reihenfolge entspricht dem Test-Loader (shuffle=False)
        ax.imshow(img)
        ax.set_title(f"wahr: {CLASSES[true[i]]}, Modell: {CLASSES[pred[i]]}", fontsize=9)
    fig.tight_layout()
    fig.savefig(f"{OUT}/misclassified.png", dpi=150)
    plt.close(fig)

# ---------------------------------------------------------------- Ergebnistabelle
lines = [
    "| Modell | Test-Accuracy | Precision | Recall | F1 | Trainierbare Parameter | Trainingszeit |",
    "|---|---|---|---|---|---|---|",
]
for name, r in results.items():
    f = r["final"]
    lines.append(f"| {name} | {f['acc']:.3f} | {f['prec']:.3f} | {f['rec']:.3f} | {f['f1']:.3f} | "
                 f"{r['params']:,} | {r['seconds'] / 60:.1f} min |")
with open(f"{OUT}/results.md", "w") as fh:
    fh.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\nFertig. Dateien liegen in '{OUT}/'.")