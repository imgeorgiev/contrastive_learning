# ClearML - Example of pytorch with tensorboard>=v1.14
#

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from torch import Tensor

from clearml import Task


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def encode(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def loss(self, x, target):
        output = self.forward(x)
        return F.nll_loss(output, target).sum()


def train(model, train_loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for data, target in train_loader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        loss = model.loss(data, target)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)


def test(model, test_loader, writer, device):
    model.eval()
    test_loss = 0
    correct = 0
    encodings = []
    with torch.no_grad():
        for niter, (data, target) in enumerate(test_loader):
            data = data.to(device)
            target = target.to(device)

            # get encodings for analysis later
            encodings.append(model.encode(data).cpu())

            # now do the actual preduction
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum").data.item()
            pred = output.data.max(1)[1]  # get the index of the max log-probability
            pred = pred.eq(target.data).cpu().sum()
            writer.add_scalar("Test/Loss", pred, niter)
            correct += pred
            if niter % 100 == 0:
                writer.add_image("test", data[0, :, :, :], niter)

    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )

    # combine all batches into one big vector
    encodings = torch.vstack(encodings)
    return encodings


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="PyTorch MNIST Example")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        metavar="N",
        help="input batch size for training (default: 64)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        metavar="LR",
        help="learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="how many batches to wait before logging training status",
    )
    args = parser.parse_args()

    # Connecting ClearML with the current process,
    # from here on everything is logged automatically
    task = Task.init(project_name="contrastive", task_name="normal")  # noqa: F841

    writer = SummaryWriter("runs")

    if torch.cuda.is_available():
        print("Using cuda")
        device = torch.device("cuda")
    else:
        print("Using CPU")
        device = torch.device("cpu")

    torch.manual_seed(args.seed)

    # set up datasets
    kwargs = {"num_workers": 4, "pin_memory": True}
    tfs = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    train_loader = DataLoader(
        datasets.MNIST("data", train=True, download=True, transform=tfs),
        batch_size=args.batch_size,
        shuffle=True,
        **kwargs,
    )
    test_loader = DataLoader(
        datasets.MNIST("data", train=False, transform=tfs),
        batch_size=args.batch_size,
        shuffle=False,
        **kwargs,
    )

    # Create model
    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Train
    for epoch in range(1, args.epochs + 1):
        loss = train(model, train_loader, optimizer, device)
        writer.add_scalar("Train/Loss", loss, epoch)
        print("Epoch: {:}/{:}, train loss: {:.2f}".format(epoch, args.epochs, loss))
    torch.save(model, "mnist_net.pkl")

    # Now test and store encodings
    encodings: Tensor = test(model, test_loader, writer, device)
    encodings = encodings.numpy()
    np.save("encodings", encodings)


if __name__ == "__main__":
    # Hack for supporting Windows OS - https://pytorch.org/docs/stable/notes/windows.html#usage-multiprocessing
    main()
