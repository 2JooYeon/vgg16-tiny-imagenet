import torch
import time
import torch.nn as nn
import torchvision.transforms as transforms
from model import VGGNET
from dataset import TinyImageNetDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('./log/')


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. config
    batch_size = 50
    epochs = 100
    lr = 0.001
    class_num = 10
    input_shape = (224, 224, 3)
    checkpoint_path = './checkpoint/'

    # 2. dataset
    train_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(),
        transforms.ToTensor(),
        transforms.Normalize([0.476, 0.431, 0.353], [0.191, 0.182, 0.174])

    ])
    train_dataset = TinyImageNetDataset(src_path='./train_10class/', input_shape=input_shape, class_num=class_num, transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.476, 0.431, 0.353], [0.191, 0.182, 0.174])
    ])

    test_dataset = TinyImageNetDataset(src_path='./test_10class/', input_shape=input_shape, class_num=class_num,  transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

    # 3. model
    model = VGGNET(in_channels=3, num_classes=class_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(lr=lr, weight_decay=0.0005, params=model.parameters(), momentum=0.9)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, verbose=True)
    model = torch.nn.parallel.DataParallel(model, device_ids=[0, ])
    model.summary()
    for epoch in range(epochs):
        start_time = time.time()
        train_loss = 0
        train_correct = 0
        train_samples = 0
        for batch_idx, (images, labels) in enumerate(train_dataloader):
            images = images.to(device=device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, preds = torch.max(outputs, 1)
            train_loss += loss
            train_correct += (preds == labels).sum().item()
            train_samples += labels.size(0)

        epoch_loss = train_loss / len(train_dataloader)
        epoch_acc = float(train_correct) / float(train_samples)
        end_time = time.time()
        print(f"train epoch {epoch + 1} | loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}, time: {end_time-start_time:.4f}")
        writer.add_scalar("Loss/train", epoch_loss, epoch)
        writer.add_scalar("Accuracy/train", epoch_acc, epoch)
        # scheduler.step(loss)
        with torch.no_grad():
            test_loss = 0
            test_correct = 0
            test_samples = 0
            start_time = time.time()
            for idx, (images, labels) in enumerate(test_dataloader):
                images = images.to(device=device)
                labels = labels.to(device=device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                _, preds = torch.max(outputs, 1)
                test_loss += loss
                test_correct += (preds == labels).sum()
                test_samples += labels.size(0)
            epoch_loss = test_loss / len(test_dataloader)
            epoch_acc = float(test_correct)/float(test_samples)
            end_time = time.time()
            print(f"test epoch {epoch + 1} | loss: {epoch_loss:.4f}, accuracy: {epoch_acc:.4f}, time: {end_time-start_time:.4f}")
            writer.add_scalar("Loss/test", epoch_loss, epoch)
            writer.add_scalar("Accuracy/test", epoch_acc, epoch)
        if epoch%9==0:
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict,': optimizer.state_dict(),
                'loss': loss
            }
            torch.save(state, checkpoint_path + 'model_{}.pth'.format(epoch))
    writer.close()
