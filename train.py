import pdb
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.optim.lr_scheduler import MultiStepLR

from torchvision.utils import make_grid
from torchvision import datasets, transforms

from util.misc import CSVLogger
from util.cutout import Cutout
from util.cutout_intensity import Cutout_intensity
from util.Cutout_Shape import Cutout_Shape
from util.cutout_intensity_shapes import Cutout_intensity_shapes

from model.resnet import ResNet18

import matplotlib.pyplot as plt

model_options = ['resnet18']
dataset_options = ['cifar10']
cutout_options = ['None', 'Cutout', 'Cutout_intesity', 'Cutout_Shape', 'Cutout_intensity_shapes']
shape_options = ['square', 'circle', 'triangle']

parser = argparse.ArgumentParser(description='CNN')
parser.add_argument('--dataset', '-d', default='cifar10',
                    choices=dataset_options)
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=False,
                    help='augment data by flipping and cropping')
parser.add_argument('--cutout', default='None', choices=cutout_options,
                    help='apply cutout')
parser.add_argument('--shape', default='square', choices=shape_options,
                    help='shape of the cutout')
parser.add_argument('--n_holes', type=int, default=1,
                    help='number of holes to cut out from image')
parser.add_argument('--length', type=int, default=16,
                    help='length of the holes')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
cudnn.benchmark = True  # Should make training should go faster for large models

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

test_id = args.dataset + '_' + args.model

print(args)

# Image Preprocessing
normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                 std=[x / 255.0 for x in [63.0, 62.1, 66.7]])

train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())

    # those who got the best acc are RandomRotation (0.414 ) and ColorJitter (0.378) for one epoch:

    # train_transform.transforms.append(transforms.RandomVerticalFlip())
    # train_transform.transforms.append(transforms.Grayscale(3))
    # train_transform.transforms.append(transforms.RandomRotation(10)) good acc for 1 epoch
    # train_transform.transforms.append(transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1))

train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

if args.cutout == 'Cutout':
    train_transform.transforms.append(Cutout(n_holes=args.n_holes, length=args.length, shape=args.shape))
if args.cutout == 'Cutout_intesity':
    train_transform.transforms.append(Cutout_intensity(n_holes=args.n_holes, length=args.length, shape=args.shape))
if args.cutout == 'Cutout_Shape':
    train_transform.transforms.append(Cutout_Shape(n_holes=args.n_holes, length=args.length, shape=args.shape))
if args.cutout == 'Cutout_intensity_shapes':
    train_transform.transforms.append(
        Cutout_intensity_shapes(n_holes=args.n_holes, length=args.length, shape=args.shape))

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

# for the 'cifar10' dataset:
num_classes = 10
train_dataset = datasets.CIFAR10(root='data/',
                                 train=True,
                                 transform=train_transform,
                                 download=True)

test_dataset = datasets.CIFAR10(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)

# Data Loader (Input Pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=args.batch_size,
                                           shuffle=True,
                                           pin_memory=True,
                                           num_workers=2)

# Print the size of the training dataset after augmentation
print("Size of training dataset after augmentation:", len(train_loader.dataset))

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=args.batch_size,
                                          shuffle=False,
                                          pin_memory=True,
                                          num_workers=2)

# Print the size of the test dataset
print("Size of test dataset:", len(test_loader.dataset))

# imply the 'resnet18':
cnn = ResNet18(num_classes=num_classes)

cnn = cnn.cuda()
criterion = nn.CrossEntropyLoss().cuda()
##################### !!!!!!!!!!!!!!1 can shift arguments ######
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

scheduler = MultiStepLR(cnn_optimizer, milestones=[60, 120, 160], gamma=0.2)

filename = 'logs/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'test_acc'], filename=filename)


def test(loader):
    cnn.eval()  # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.cuda()
        labels = labels.cuda()

        with torch.no_grad():
            pred = cnn(images)

        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()

    return val_acc


for epoch in range(args.epochs):

    xentropy_loss_avg = 0.
    correct = 0.
    total = 0.

    progress_bar = tqdm(train_loader)

    for i, (images, labels) in enumerate(progress_bar):
        progress_bar.set_description('Epoch ' + str(epoch))

        images = images.cuda()
        labels = labels.cuda()

        cnn.zero_grad()
        pred = cnn(images)

        xentropy_loss = criterion(pred, labels)
        xentropy_loss.backward()
        cnn_optimizer.step()

        xentropy_loss_avg += xentropy_loss.item()

        # Calculate running average of accuracy
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels.data).sum().item()
        accuracy = correct / total

        progress_bar.set_postfix(
            xentropy='%.3f' % (xentropy_loss_avg / (i + 1)),
            acc='%.3f' % accuracy)

        # Visualize the first image in the batch after normalization
        if i == 44 and epoch == 0:
            image = images[0].cpu().numpy()
            image = np.transpose(image, (1, 2, 0))  # Transpose (C, H, W) to (H, W, C)
            image = image * [x / 255.0 for x in [63.0, 62.1, 66.7]] + [x / 255.0 for x in [125.3, 123.0, 113.9]]
            image = np.clip(image, 0, 1)  # Clip values to [0, 1] range

            plt.imshow(image)
            plt.title(f"Epoch: {epoch}, Label: {labels[0]}")
            plt.axis('off')

            # Save the image instead of displaying it interactively
            plt.savefig(f"image_epoch_{epoch}_label_{labels[0]}.png")
            plt.close()  # Close the plot to free up memory

    test_acc = test(test_loader)
    tqdm.write('test_acc: %.3f' % (test_acc))

    scheduler.step(epoch)  # Use this line for PyTorch <1.4
    # scheduler.step()     # Use this line for PyTorch >=1.4

    row = {'epoch': str(epoch), 'train_acc': str(accuracy), 'test_acc': str(test_acc)}
    csv_logger.writerow(row)

torch.save(cnn.state_dict(), 'checkpoints/' + test_id + '.pt')
csv_logger.close()
