import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm # 导入 tqdm
from torch.utils.tensorboard import SummaryWriter # 导入 SummaryWriter

from capsule import CapsuleNet
from dataset import get_mnist_loaders

# 假设 device 已经定义
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 修改后的 train 函数 ---
def train(model, train_loader, optimizer, criterion, epoch, writer):
    model.train()
    
    # 使用 tqdm 包装 train_loader，并添加描述
    loop = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (data, target) in enumerate(loop):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        # --- 计算指标 ---
        v_length = torch.sqrt((output**2).sum(dim=2))
        pred = v_length.argmax(dim=1)
        correct = pred.eq(target).sum().item()
        accuracy = correct / target.size(0)
        
        # --- TensorBoard 记录 ---
        # 计算全局步数，用于在 TensorBoard 中正确显示 x 轴
        global_step = (epoch - 1) * len(train_loader) + batch_idx
        writer.add_scalar('Loss/train_batch', loss.item(), global_step)
        writer.add_scalar('Accuracy/train_batch', accuracy, global_step)

        # --- 更新 tqdm 进度条的后缀信息 ---
        loop.set_postfix(loss=loss.item(), acc=f"{accuracy:.2%}")

# --- 修改后的 test 函数 ---
def test(model, test_loader, criterion, epoch, writer):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    # 测试时也可以用 tqdm，但通常信息较少，可以不用
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            total_loss += criterion(output, target).item()
            
            v_length = torch.sqrt((output**2).sum(dim=2))
            pred = v_length.argmax(dim=1)
            total += target.size(0)
            correct += pred.eq(target).sum().item()

    avg_loss = total_loss / len(test_loader)
    total_accuracy = correct / total

    print(f"\nTest set: Average loss: {avg_loss:.4f}, Accuracy: {correct}/{total} ({total_accuracy:.2%})\n")
    
    # --- TensorBoard 记录整个 epoch 的测试结果 ---
    writer.add_scalar('Loss/test_epoch', avg_loss, epoch)
    writer.add_scalar('Accuracy/test_epoch', total_accuracy, epoch)

    return total_accuracy

# --- 修改后的 main 函数 ---
def main():
    # 1. 初始化 SummaryWriter，日志会保存在 'runs/capsulenet_experiment_1' 文件夹
    writer = SummaryWriter('runs/capsulenet_experiment_1')

    train_loader, test_loader = get_mnist_loaders(batch_size=128)

    model = CapsuleNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = StepLR(optimizer, step_size=5, gamma=0.95)
    criterion = model.margin_loss

    best_accuracy = 0.0
    num_epochs = 50

    for epoch in range(1, num_epochs + 1):
        # 将 writer 传入 train 和 test 函数
        train(model, train_loader, optimizer, criterion, epoch, writer)
        current_accuracy = test(model, test_loader, criterion, epoch, writer)
        scheduler.step()

        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            print(f"🚀 New best accuracy: {best_accuracy:.2%}. Saving model...")
            torch.save(model.state_dict(), 'best_model.pth')
    
    # 3. 训练结束后关闭 writer
    writer.close()
    print("Training finished. TensorBoard logs saved.")

if __name__ == '__main__':
    main()