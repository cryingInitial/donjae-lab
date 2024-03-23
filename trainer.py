import time
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import torch
from torch import nn, optim
from tqdm import tqdm
from backbone import get_model
import torch.nn.functional as F
import matplotlib.pyplot as plt


def loss_picker(loss):
    if loss == 'mse':
        criterion = nn.MSELoss()
    elif loss == 'cross':
        criterion = nn.CrossEntropyLoss()
    else:
        print("automatically assign mse loss function to you...")
        criterion = nn.MSELoss()

    return criterion


def optimizer_picker(optimization, param, lr, momentum=0.):
    if optimization == 'adam':
        optimizer = optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = optim.SGD(param, lr=lr, momentum=momentum)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = optim.Adam(param, lr=lr)
    return optimizer

def train_and_save(model, train_loader, test_loader, model_name, args, mode='ori'):
    print("MODE: ", mode)
    model = model.to(args.device)
    model.train()
    criterion = nn.CrossEntropyLoss()
    if args.data_name == 'cifar10':
        optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    elif args.data_name == 'cifar100':
        optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=280)
    
    for epoch in range(1, args.train_epochs + 1):
        running_loss = train(model, train_loader, criterion, optimizer, "cross", args.device)
        acc = test(model, test_loader, args.num_classes)
        print(f'epoch {epoch} loss: {running_loss:.4f}, acc: {acc}')
        scheduler.step()

    torch.save(model, f'./checkpoints/{model_name}_{mode}.pth')

def train(model, data_loader, criterion, optimizer, loss_mode, device='cpu'):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):
        # print(batch_y.size())

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        optimizer.zero_grad()
        output = model(batch_x)  # get predict label of batch_x

        if loss_mode == "mse":
            loss = criterion(output, batch_y)  # mse loss
        elif loss_mode == "cross":
            loss = criterion(output, batch_y)  # torch.argmax(batch_y, dim=1))  # cross entropy loss
        elif loss_mode == 'neg_grad':
            loss = -criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return running_loss

def test(model, loader, class_num):
    model.eval()
    outputavg = [0.] * class_num
    cnt = [0] * class_num
    res = ''
    with torch.no_grad():
        for idx, (data, target) in enumerate(tqdm(loader, leave=False)):
            # target = target.item()

            data = data.cuda()
            target = target.cuda()

            output = model(data)
            output = F.softmax(output, dim=-1).data.cpu().numpy().tolist()
            for i in range(len(target)):
                pred = target[i].cpu().int()
                if round(output[i][pred]) == 1:
                    outputavg[pred] += 1
                cnt[pred] += 1

    for i in range(len(outputavg)):
        if cnt[i] == 0:
            outputavg[i] = 0.
        else:
            outputavg[i] /= cnt[i]
        res += 'class {} acc: {:.2%}\n'.format(i, outputavg[i])
    res += 'avg acc: {:.2%}'.format(sum(outputavg) / class_num)
    return res


def eval(model, data_loader, batch_size=64, mode='backdoor', print_perform=False, device='cpu', name=''):
    model.eval()  # switch to eval status

    y_true = []
    y_predict = []
    for step, (batch_x, batch_y) in enumerate(data_loader):

        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        batch_y_predict = model(batch_x)
        if mode == 'pruned':
            batch_y_predict = batch_y_predict[:, 0:10]
        elif type(mode) == int:
            batch_y_predicts = torch.chunk(batch_y_predict, mode, dim=-1)
            batch_y_predict = sum(batch_y_predicts) / mode


        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        # batch_y = torch.argmax(batch_y, dim=1)
        y_predict.append(batch_y_predict)
        y_true.append(batch_y)

    y_true = torch.cat(y_true, 0)
    y_predict = torch.cat(y_predict, 0)

    num_hits = (y_true == y_predict).float().sum()
    acc = num_hits / y_true.shape[0]
    # print()

    if print_perform and mode != 'backdoor' and mode != 'widen' and mode != 'pruned':
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
    if print_perform and mode == 'widen':
        class_name = data_loader.dataset.classes.append('extra class')
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=class_name, digits=4))
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.show()
    if print_perform and mode == 'pruned':
        # print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes, digits=4))
        class_name = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]#['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        C = confusion_matrix(y_true.cpu(), y_predict.cpu(), labels=class_name)
        plt.matshow(C, cmap=plt.cm.Reds)
        plt.ylabel('True Label')
        plt.xlabel('Pred Label')
        plt.title('{} confusion matrix'.format(name), loc='center')
        plt.show()

    return accuracy_score(y_true.cpu(), y_predict.cpu()), acc
