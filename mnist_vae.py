import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.distributions as pyd

from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()

from IPython.core import ultratb
import sys

# For debugging
sys.excepthook = ultratb.FormattedTB(mode="Plain", color_scheme="Neutral", call_pdb=1)


class Autoencoder(nn.Module):
    def __init__(self, hidden_dim):
        super(Autoencoder, self).__init__()
        # input is 28x28 images
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ELU(),
            nn.Flatten(1, -1),
            nn.Linear(320, 50),
            nn.ELU(),
            nn.Linear(50, 2 * hidden_dim),
            nn.ELU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 50),
            nn.ELU(),
            nn.Linear(50, 320),
            nn.ELU(),
            nn.Unflatten(1, (20, 4, 4)),
            nn.ConvTranspose2d(20, 10, kernel_size=5, stride=2, output_padding=1),
            nn.ELU(),
            nn.ConvTranspose2d(10, 1, kernel_size=5, stride=2, output_padding=1),
            nn.ELU(),
        )
        self.hidden_dim = hidden_dim

    def forward(self, x):
        mu, logstd = self.encoder(x).chunk(2, dim=-1)
        dist = pyd.Normal(mu, logstd.exp())
        z = dist.rsample()
        x = self.decoder(z)
        return x, mu, logstd

    def encode(self, x):
        mu, logstd = self.encoder(x).chunk(2, dim=-1)
        dist = pyd.Normal(mu, logstd.exp())
        return dist.rsample()


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = Autoencoder(10).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_data = MNIST(root="./data", train=True, transform=tf, download=True)
    train_loader = DataLoader(
        train_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4
    )
    test_data = MNIST(root="./data", train=False, transform=tf)
    test_loader = DataLoader(
        test_data, batch_size=cfg.batch_size, shuffle=False, num_workers=4
    )

    for i in range(cfg.epochs):
        total_loss = 0.0
        for data, labels in tqdm(train_loader):
            data = data.to(device)
            labels = labels.to(device)
            opt.zero_grad()
            output, mu, logstd = model(data)
            if cfg.loss == "mse":
                loss = F.mse_loss(output, data)
                kld = 0.5 * torch.sum(logstd + 1 - logstd.exp() - mu**2, dim=1)
                loss -= cfg.kld_loss_coeff * torch.mean(kld)
            elif cfg.loss == "contrastive":
                # contrastive loss
                dim = labels.shape[0]
                mask = labels.repeat(dim) == torch.repeat_interleave(labels, dim)
                dists = torch.norm(
                    z.repeat((dim, 1)) - torch.repeat_interleave(z, dim, dim=0),
                    dim=1,
                )
                # TODO not sure if correct for this to be the only loss
                loss = cfg.contrastive_loss_coeff * torch.mean(
                    mask * dists + (~mask) * torch.clamp(cfg.margin - dists, min=0) ** 2
                )
            loss.backward()
            opt.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        with torch.no_grad():
            test_loss = 0.0
            for data, _ in tqdm(test_loader):
                output, _, _ = model(data.to(device))
                loss = F.mse_loss(output, data.to(device))
                test_loss += loss.item()
            test_loss /= len(test_loader)
        print(
            f" Epoch {i}/{cfg.epochs}  Train loss: {total_loss:.2f} Test loss: {test_loss:.2f}"
        )

    torch.save(model, f"mnist_vae_{cfg.loss}.pt")

    # collect encodings for analysis
    print("Collecting encodings")
    encodings = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            encodings.append(model.encode(data.to(device)).detach().cpu())
    encodings = torch.vstack(encodings)
    encodings = np.array(encodings)
    np.save(f"encodings_vae_{cfg.loss}.npy", encodings)


if __name__ == "__main__":
    train()
