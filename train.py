import os
from random import random

import numpy as np
import torch
import torchvision.transforms
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from model import AudioDFNet
from dataset import ASVspoofDataset
from torchvision.datasets import CIFAR10
import shutil

# recreate tensorboard log
# shutil.rmtree('./log')
# os.mkdir('./log')


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# set a random seed
setup_seed(20)

# initializing hyperparameters
epoch_num = 50
batch_size = 16
total_train_step = 0
total_dev_step = 0

# train_set = ASVspoofDataset(train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                                                   torchvision.transforms.Normalize(
#                                                                                       (0.5,), (0.5,)),
#                                                                                   torchvision.transforms.RandomCrop(
#                                                                                       (63, 13))]))
# dev_set = ASVspoofDataset(train=False, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
#                                                                                  torchvision.transforms.Normalize(
#                                                                                      (0.5,), (0.5,)),
#                                                                                  torchvision.transforms.RandomCrop(
#                                                                                      (68, 13))]))

train_set = ASVspoofDataset(train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                  torchvision.transforms.Resize(
                                                                                      (32, 32))]))
dev_set = ASVspoofDataset(train=True, transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                                                torchvision.transforms.Resize(
                                                                                    (32, 32))]))

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=True, drop_last=True)

writer = SummaryWriter('./log')

df_net = AudioDFNet()
if torch.cuda.is_available():
    df_net = df_net.cuda()

loss_fn = nn.CrossEntropyLoss()
if torch.cuda.is_available():
    loss_fn = loss_fn.cuda()

learning_rate = 1e-3
optimizer = torch.optim.Adam(df_net.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08)
# optimizer = torch.optim.SGD(df_net.parameters(), lr=learning_rate)

for epoch in range(epoch_num):
    print("epoch {} start to train...".format(epoch + 1))

    # train
    df_net.train()

    for data in train_loader:
        mats, labels = data
        if torch.cuda.is_available():
            mats = mats.cuda()
            labels = labels.cuda()
        outputs = df_net(mats)
        loss = loss_fn(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step += 1

        if total_train_step % 10 == 0:
            print("train session: {}, loss: {}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss", loss.item(), total_train_step)

    # eval
    df_net.eval()
    total_dev_loss = 0
    total_accuracy = 0

    with torch.no_grad():
        for data in dev_loader:
            mats, labels = data
            if torch.cuda.is_available():
                mats = mats.cuda()
                labels = labels.cuda()
            outputs = df_net(mats)
            loss = loss_fn(outputs, labels)
            total_dev_loss += loss.item()
            accuracy = (outputs.argmax(1) == labels).sum()
            total_accuracy += accuracy

    print("dev loss: {}".format(total_dev_loss))
    print("dev accuracy: {}".format(total_accuracy / dev_set.__len__()))
    total_dev_step += 1
    writer.add_scalar("dev loss", total_dev_loss, total_dev_step)
    writer.add_scalar("dev accuracy", total_accuracy / dev_set.__len__(), total_dev_step)

    torch.save(df_net, './model2/AudioDFNet{}.pt'.format(epoch))

print("train over")
writer.close()
