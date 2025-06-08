from capsule import CapsuleNet
from dataset import get_mnist_loaders
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

    print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader)}, Accuracy: {100. * correct / total}')


def main():
    train_loader, test_loader = get_mnist_loaders(batch_size=32)

    model = CapsuleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = model.margin_loss

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch)

if __name__ == "__main__":
    main()

