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

        v_length = torch.sqrt((output**2).sum(dim=2))  # (batch_size, num_caps)
        pred = v_length.argmax(dim=1)  # (batch_size,)
        
        total_loss += loss.item()
        total += target.size(0)
        correct += pred.eq(target).sum().item()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    print(f'Epoch: {epoch}, Loss: {total_loss / len(train_loader)}, Accuracy: {100. * correct / total:.2f}%')


def main():
    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    model = CapsuleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = model.margin_loss

    num_epochs = 50
    for epoch in range(1, num_epochs + 1):
        train(model, train_loader, optimizer, criterion, epoch)

if __name__ == "__main__":
    main()

