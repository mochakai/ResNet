import os
import sys
from argparse import ArgumentParser
from datetime import datetime
import json

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision.models as models

from dataloader import RetinopathyLoader


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class BasicBlock(nn.Module):
    out_extend = 1
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride,
                     padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3,
                     padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.downsample = downsample

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = self.relu(out)
        return out

class Bottleneck(nn.Module):
    out_extend = 4
    def __init__(self, in_ch, out_ch, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.conv3 = nn.Conv2d(out_ch, out_ch * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)

        residual = self.downsample(x) if self.downsample is not None else x
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, out_class=5, in_ch=3):
        self.input_size = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._customize_layer(block, 64, layers[0])
        self.layer2 = self._customize_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._customize_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._customize_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(51200 * block.out_extend, out_class)

    def _customize_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.input_size != planes * block.out_extend:
            downsample = nn.Sequential(
                nn.Conv2d(self.input_size, planes * block.out_extend, 
                            kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.out_extend),
            )
        layers = []
        layers.append(block(self.input_size, planes, stride, downsample))
        self.input_size = planes * block.out_extend
        for i in range(1, blocks):
            layers.append(block(self.input_size, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        res = x.view(x.size(0), -1)
        out = self.fc(res)
        return out


def test_accuracy(net, dataset):
    net.eval()
    test_size = dataset.data_size
    # test_size = 100
    accuracy_res = []
    pred_list = []
    gt_list = []
    for index in range(test_size):
        x, gt = dataset[index]
        t_x = torch.unsqueeze(torch.tensor(x), dim=0).type(torch.FloatTensor).to(device)
        pred_y = net(t_x)
        pred_y = torch.max(pred_y.cpu(), 1)[1].data.numpy()[0]
        accuracy_res.append(pred_y == gt)
        pred_list.append(int(pred_y))
        gt_list.append(int(gt))

    accuracy = float(np.array(accuracy_res).astype(int).sum()) / float(test_size)
    net.train()
    return accuracy * 100, pred_list, gt_list


def train_accuracy(pred_y, gt):
    gt = np.array(gt)
    pred_y = np.array(pred_y)
    accuracy = float((pred_y == gt).astype(int).sum()) / float(len(gt))
    return accuracy * 100


def handle_param(args):
    if args.model == 'resnet18':
        if args.untrained:
            net = ResNet(BasicBlock, [2, 2, 2, 2])
        else:
            net = models.resnet18(pretrained=True)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.fc = nn.Linear(512, 5)
    elif args.model == 'resnet50':
        if args.untrained:
            net = ResNet(Bottleneck, [3, 4, 6, 3])
        else:
            net = models.resnet50(pretrained=True)
            net.avgpool = nn.AdaptiveAvgPool2d(1)
            net.fc = nn.Linear(512 * 4, 5)
    else:
        raise TypeError('Error: {} model type not defined.'.format(args.model))

    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
                        net.parameters(), lr=args.learning_rate, 
                        weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsp':
        optimizer = torch.optim.RMSprop(
                        net.parameters(), lr=args.learning_rate, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
                        net.parameters(), lr=args.learning_rate, 
                        momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        raise TypeError('Error: {} optimizer type not defined.'.format(args.optimizer))

    if args.loss_function == 'CrossEntropy':
        loss_function = nn.CrossEntropyLoss()
    else:
        raise TypeError('Error: {} loss_function type not defined.'.format(args.loss_function))
    return net.to(device), optimizer, loss_function


def main(args):
    train_data = RetinopathyLoader('data', 'train')
    test_data = RetinopathyLoader('data', 'test')
    train_loader = Data.DataLoader(dataset=train_data, batch_size=args.batch, shuffle=True)
    net, optimizer, loss_function = handle_param(args)
    full_step = ((train_data.data_size // args.batch) +\
                ((train_data.data_size % args.batch)>0)) * args.epochs

    train_tag = '_untrained' if args.untrained else '_pretrained'
    file_name = args.model + train_tag + '_' +\
                args.optimizer + '_ep' + str(args.epochs) +\
                '_lr' + str(args.learning_rate) + '_wd' + str(args.weight_decay)
    if args.load:
        net.load_state_dict(torch.load(args.load))
        test_acc, pred, gt = test_accuracy(net, test_data)

        with open(args.model + train_tag + '.json', 'w') as f:
            json.dump({
                'pred_y': pred,
                'gt': gt,
                'class': list(range(5)),
            }, f)
        print('test_acc: {:.4f} %'.format(test_acc))
        return
    # start training
    max_acc = 0
    count = 0
    acc_dict = {'train'+train_tag: [], 'test'+train_tag: []}

    print(datetime.now().strftime("%m-%d %H:%M"), 'start training...')
    for epoch in range(args.epochs):
        print('-'*10, 'epoch', epoch+1, '-'*10)
        y_list, gt_list = [], []
        for b_x, b_y in train_loader:
            count += 1
            b_x = b_x.type(torch.FloatTensor).to(device)
            b_y = b_y.to(device)
            output = net(b_x)
            loss = loss_function(output, b_y.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            y_list.extend(torch.max(output, 1)[1].data.tolist())
            gt_list.extend(b_y.data.tolist())
            if count % args.step == 0:
                print(datetime.now().strftime("%m-%d %H:%M"))
                print('({} / {}) loss: {:.4f} | train_acc: {:.2f} %'.format(
                        count, full_step, loss, train_accuracy(y_list, gt_list)))

        test_acc, _, _ = test_accuracy(net, test_data)
        if test_acc > max_acc:
            max_acc = test_acc
            torch.save(net.state_dict(), file_name + '.pkl')
        acc_dict['train'+train_tag].append(train_accuracy(y_list, gt_list))
        acc_dict['test'+train_tag].append(test_acc)
        print(datetime.now().strftime("%m-%d %H:%M"))
        print('({} / {}) train_acc: {:.2f} % | test_acc: {:.2f} %'.format(
                count, full_step, train_accuracy(y_list, gt_list), test_acc))

    with open(file_name + '.json', 'w') as f:
        json.dump({
            'x': list(range(args.epochs)),
            'y_dict': acc_dict,
            'title': args.model + train_tag,
        }, f)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("-s", "--step", help="when to print log", type=int, default=1000)
    parser.add_argument("-b", "--batch", help="batch size", type=int, default=4)
    parser.add_argument("-ep", "--epochs", help="your training target", type=int, default=10)
    parser.add_argument("-m", "--model", help="resnet18 | resnet50", type=str, default='resnet18')
    parser.add_argument("-ut", "--untrained", help="model pretrained(default)", action='store_true')
    parser.add_argument("-opt", "--optimizer", help="adam | rmsp | sgd", type=str, default='sgd')
    parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=1e-3)
    parser.add_argument("-wd", "--weight_decay", help="weight decay", type=float, default=5e-4)
    parser.add_argument("-mom", "--momentum", help="momentum factor", type=float, default=0.9)
    parser.add_argument("-lf", "--loss-function", help="loss function", type=str, default='CrossEntropy')
    parser.add_argument("-load", "--load", help="your pkl file path", type=str, default='')
    return parser.parse_args()


if __name__ == "__main__":
    main(get_args())

    sys.stdout.flush()
    sys.exit()