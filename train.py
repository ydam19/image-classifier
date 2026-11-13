"""CIFAR-10 이미지 분류: ImageNet 사전학습 ResNet18 전이학습"""
import json
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms

# ---------------- 설정 ----------------
EPOCHS = 3
BATCH_SIZE = 64
LR = 1e-4
SEED = 42

torch.manual_seed(SEED)
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"device: {device}")

# ---------------- 데이터 ----------------
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]  # ImageNet 정규화 값
tf_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])
tf_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

train_ds = datasets.CIFAR10("data", train=True, download=True, transform=tf_train)
test_ds = datasets.CIFAR10("data", train=False, download=True, transform=tf_test)
train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
test_dl = torch.utils.data.DataLoader(test_ds, batch_size=128, num_workers=0)
classes = train_ds.classes

# ---------------- 모델 ----------------
model = models.resnet18(weights="IMAGENET1K_V1")
model.fc = nn.Linear(model.fc.in_features, len(classes))  # 마지막 층만 10개 클래스로 교체
model = model.to(device)

opt = torch.optim.Adam(model.parameters(), lr=LR)
loss_fn = nn.CrossEntropyLoss()


def evaluate():
    model.eval()
    correct = torch.zeros(len(classes))
    total = torch.zeros(len(classes))
    with torch.no_grad():
        for x, y in test_dl:
            pred = model(x.to(device)).argmax(1).cpu()
            for c in range(len(classes)):
                mask = y == c
                total[c] += mask.sum()
                correct[c] += (pred[mask] == c).sum()
    return (correct.sum() / total.sum()).item(), (correct / total).tolist()


# ---------------- 학습 ----------------
history = []
start = time.time()
for epoch in range(1, EPOCHS + 1):
    model.train()
    total_loss, n_correct, n_seen = 0.0, 0, 0
    t0 = time.time()
    for i, (x, y) in enumerate(train_dl, 1):
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        loss.backward()
        opt.step()
        total_loss += loss.item()
        n_correct += (out.argmax(1) == y).sum().item()
        n_seen += y.size(0)
        if i % 100 == 0:
            print(f"  epoch {epoch} step {i}/{len(train_dl)} loss {total_loss / i:.4f}", flush=True)

    test_acc, class_acc = evaluate()
    rec = {
        "epoch": epoch,
        "train_loss": round(total_loss / len(train_dl), 4),
        "train_acc": round(n_correct / n_seen, 4),
        "test_acc": round(test_acc, 4),
        "seconds": round(time.time() - t0),
    }
    history.append(rec)
    print(rec, flush=True)

# ---------------- 결과 저장 ----------------
torch.save(model.state_dict(), "model.pt")
results = {
    "config": {"model": "resnet18 (IMAGENET1K_V1)", "epochs": EPOCHS, "batch_size": BATCH_SIZE,
               "lr": LR, "optimizer": "Adam", "device": device, "seed": SEED},
    "history": history,
    "final_test_acc": history[-1]["test_acc"],
    "class_acc": {c: round(a, 4) for c, a in zip(classes, class_acc)},
    "total_minutes": round((time.time() - start) / 60, 1),
}
json.dump(results, open("results.json", "w"), indent=2)

ep = [h["epoch"] for h in history]
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(ep, [h["train_loss"] for h in history], marker="o")
ax[0].set(title="Train Loss", xlabel="Epoch", xticks=ep)
ax[1].plot(ep, [h["train_acc"] for h in history], marker="o", label="train")
ax[1].plot(ep, [h["test_acc"] for h in history], marker="o", label="test")
ax[1].set(title="Accuracy", xlabel="Epoch", xticks=ep)
ax[1].legend()
plt.tight_layout()
plt.savefig("training_curve.png", dpi=120)
print("done:", results["final_test_acc"], f"({results['total_minutes']} min)")
