import torch
import torchvision.datasets
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import torch.optim as optim
from convmixer import ConvMixer
import argparse
import os
import time

parser = argparse.ArgumentParser(description='train ConvMixer on CIFARs')
#Train params
parser.add_argument('--dataset', type=str, choices=['cifar100', 'cifar10'], default='cifar100')
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
#Opt params
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--milestones', type=int, nargs='+', default=[75, 125, 175])
#Model Params
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--depth', type=int, default=8)
parser.add_argument('--kernel-size', type=int, default=9)
parser.add_argument('--patch-size', type=int, default=1)
#Path Params
parser.add_argument('--data', type=str, default='./data')
parser.add_argument('--save', type=str, default='./outputs')
#Evaluation Params
parser.add_argument('--test', action='store_true')

def load_data(args):
    os.makedirs(args.save, exist_ok=True)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))])

    if args.dataset == 'cifar10':
        train_set = torchvision.datasets.CIFAR10(root=args.data, train=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR10(root=args.data, train=False, transform=test_transform)
        num_classes = 10
    elif args.dataset == 'cifar100':
        num_classes = 100
        train_set = torchvision.datasets.CIFAR100(root=args.data, train=True, transform=train_transform)
        test_set = torchvision.datasets.CIFAR100(root=args.data, train=False, transform=test_transform)

    return train_set, test_set, num_classes


def train(args):
    train_set, test_set, num_classes = load_data(args)

    model = ConvMixer(dim=args.dim, depth=args.depth, patch_size=args.patch_size, kernel_size=args.kernel_size, n_classes=num_classes).cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=False)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)

    best_acc = 0.
    for i in range(args.epochs):
        running_loss = 0.
        model.train()
        t = time.process_time()
        for idx, (images, labels) in enumerate(train_loader):

            optimizer.zero_grad()

            images = images.cuda()
            labels = labels.cuda()

            output = model(images)
            loss = F.cross_entropy(output, labels)
            running_loss = running_loss * 0.1 + 0.9 * loss.item()

            loss.backward()
            optimizer.step()
        scheduler.step()
        elapsed_time = time.process_time() - t

        val_acc = accuracy(model, val_loader)
        print(f'Epoch: {i}/{args.epochs}, time: {elapsed_time:.3f}, loss:{running_loss:.4f}, val_acc: {val_acc}, best_acc:{best_acc}')
        if val_acc > best_acc:
            torch.save(model.state_dict(), os.path.join(args.save, 'best.pth'))


def accuracy(model, val_loader):
    total = 0
    correct = 0
    with torch.no_grad():
        model.eval()
        for idx, (images, labels) in enumerate(val_loader):
            outputs = model(images.cuda())
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels.cuda()).sum().item()

    return 100.0 * correct / total


def test(args):

    _, test_set, num_classes = load_data(args)
    val_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=8, pin_memory=False)
    model = ConvMixer(dim=args.dim, depth=args.depth, patch_size=args.patch_size, kernel_size=args.kernel_size, n_classes=num_classes).cuda()

    val_acc = accuracy(model, val_loader)
    print(f'Test time accuracy: {val_acc}')


if __name__ == '__main__':
    args = parser.parse_args()
    if not args.test:
        train(args)
    test(args)
