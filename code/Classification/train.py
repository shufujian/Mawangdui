# author: alex
# time: 2023/4/22 00:31
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Dataset

from pretrained_model.DenseNet201 import *
from pretrained_model.ResNet152 import *
from pretrained_model.ShuffleNet import *
from pretrained_model.Efficient import *
from pretrained_model.MobileNet import *

def train(model_name):

    transform_train = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.RandomAffine(15, translate=(0, 0.1), scale=None, shear=2),
        transforms.ColorJitter(brightness=(0,0.2), contrast=(0,0.1), saturation=(0,0.3), hue=0),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    # transform_train = transforms.RandomApply(transform_train, p=0.7)
    transform_test = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset = ImageFolder('./data/mwd_new/train', transform=transform_train)
    val_dataset = ImageFolder('./data/mwd_new/val', transform=transform_test)
    test_dataset = ImageFolder('./data/mwd_new/test', transform=transform_test)
    # 设置超参数
    EPOCH = 20
    pre_epoch = 0
    BATCH_SIZE = 64
    LR = 0.001

    print(len(train_dataset.class_to_idx))
    print(len(train_dataset))

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=2)

    # GPU或者CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if model_name == 'EfficientNet':
        model = get_EfficientNet_pre(len(train_dataset.class_to_idx))
    elif model_name == 'EfficientNet_un':
        model = get_EfficientNet_unpre(len(train_dataset.class_to_idx))
    elif model_name == 'ResNet':
        model = get_ResNet152_pre(len(train_dataset.class_to_idx))
    elif model_name == 'ResNet_un':
        model = get_ResNet152_unpre(len(train_dataset.class_to_idx))
    elif model_name == 'DenseNet':
        model = get_DenseNet201_pre(len(train_dataset.class_to_idx))
    elif model_name == 'DenseNet_un':
        model = get_DenseNet201_unpre(len(train_dataset.class_to_idx))
    elif model_name == 'MobileNet':
        model = get_mobileNet_pre(len(train_dataset.class_to_idx))
    elif model_name == 'MobileNet_un':
        model = get_mobileNet_unpre(len(train_dataset.class_to_idx))
    elif model_name == 'ShuffleNet_un':
        model = get_ShuffleNet_unpre(len(train_dataset.class_to_idx))
    elif model_name == 'ShuffleNet':
        model = get_ShuffleNet_pre(len(train_dataset.class_to_idx))

    model = model.to(device)

    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 记录训练过程中的参数
    val_acc = []
    train_acc = []
    train_loss = []
    val_loss = []
    max_correct = 0.0

    for epoch in range(pre_epoch, EPOCH):
        print('\nEpoch: %d' % (epoch + 1))
        model.train()
        sum_loss = 0.0
        correct = 0.0
        total = 0.0
        for i, data in enumerate(train_loader, 0):
            # 预处理数据
            length = len(train_loader)
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()

            # forward & backward
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # 计算每个batch的loss和correct
            sum_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += predicted.eq(labels.data).cpu().sum()
            # print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
            #       % (epoch + 1, (i + 1 + epoch * length), sum_loss / (i + 1), 100. * correct / total))
        print('[epoch:%d] Loss: %.03f | Acc: %.3f%% '
              % (epoch + 1, sum_loss / len(train_loader.dataset), 100. * correct.item() / total))

        train_acc.append(100. * correct.item() / total)
        train_loss.append(sum_loss / len(train_loader.dataset))

        # get the ac with testdataset in each epoch
        print('Waiting Test...')
        with torch.no_grad():
            correct = 0
            sum_loss = 0.0
            total = 0.0
            for data in val_loader:
                model.eval()
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)

                loss = criterion(outputs, labels)
                sum_loss += loss.item()

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            acc = 100 * correct.item() / total

            if (max_correct < acc):
                torch.save(model.state_dict(), './model_final/' + model_name + '.pth')
                max_correct = acc
            print('Test\'s ac is: %.3f%%, Test\'s loss: %.3f, Best Acc %.3f%%' % (
                acc, sum_loss / len(val_loader.dataset), max_correct))
            val_acc.append(100. * correct.item() / total)
            val_loss.append(sum_loss / len(val_loader.dataset))

    print('Train has finished, total epoch is %d' % EPOCH)

    # 测试test上的准确率
    test_correct = 0
    test_loss = 0.0
    test_total = 0
    test_acc = 0.0

    model.load_state_dict(torch.load('./model_final/' + model_name + '.pth'))

    with torch.no_grad():
        for data in test_loader:
            model.eval()
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum()
        test_acc = 100 * test_correct.item() / test_total

        print('Test\'s ac is: %.3f%%, Test\'s loss: %.3f' % (test_acc, test_loss / len(test_loader.dataset)))

    # 统计准确率和损失值
    import pandas as pd

    d = {
        'Train_accuracy': train_acc,
        'Val_accuracy': val_acc,
        'max_accuracy': max_correct,
        'Test_accuracy': test_acc,
        'Train_loss': train_loss,
        'Val_loss': val_loss,
        'Test_loss': test_loss / len(test_loader.dataset)
    }
    dat = pd.DataFrame(d)
    dat.to_csv('./out_final/' + model_name + '_record.csv')


# model_name = ['ShuffleNet', 'MobileNet','ResNet', 'DenseNet']

model_name = ['ShuffleNet','ShuffleNet_un','EfficientNet','EfficientNet_un','MobileNet','MobileNet_un','DenseNet','ResNet','DenseNet_un','ResNet_un']

for i in model_name:
    print(i)
    train(i)