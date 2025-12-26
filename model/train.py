import torch
from torch.utils.data import DataLoader, random_split
from dataset import OASISDataset
from model import ResNet3D
import torch.nn as nn

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = OASISDataset("../data")
    total = len(dataset)
    val_size = int(0.2 * total)
    train_size = total - val_size

    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

    model = ResNet3D(num_classes=4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    epochs = 5

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        vloss = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                vloss += criterion(out, y).item()
                pred = torch.argmax(out, dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        acc = correct / total
        print("epoch:", epoch, "val_loss:", vloss, "val_acc:", acc)

if __name__ == "__main__":
    main()
