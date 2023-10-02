import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader

from tqdm import tqdm
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()


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
            nn.Linear(50, hidden_dim),
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

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


@hydra.main(config_path="cfg", config_name="config.yaml", version_base="1.2")
def train(cfg: DictConfig):
    torch.manual_seed(cfg.seed)
    device = torch.device(cfg.device)
    model = Autoencoder(10).to(device)
    opt = optim.Adam(model.parameters(), lr=cfg.learning_rate)

    tf = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    train_data = MNIST(root="./data", train=True, transform=tf)
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
            z = model.encoder(data)
            output = model.decoder(z)
            loss = F.mse_loss(output, data)
            # contrastive loss
            dim = labels.shape[0]
            mask = labels.repeat(dim) == torch.repeat_interleave(labels, dim)
            dists = (
                torch.norm(
                    z.repeat((dim, 1)) - torch.repeat_interleave(z, dim, dim=0), dim=1
                )
                ** 2
            )
            loss += cfg.contrastive_loss_coeff * torch.mean(
                mask * dists + (~mask) * torch.clamp(cfg.margin - dists, min=0) ** 2
            )
            loss.backward()
            opt.step()
            total_loss += loss.item()
        total_loss /= len(train_loader)

        with torch.no_grad():
            test_loss = 0.0
            for data, _ in tqdm(test_loader):
                output = model(data.to(device))
                loss = F.mse_loss(output, data.to(device))
                test_loss += loss.item()
            test_loss /= len(test_loader)
        print(
            f" Epoch {i}/{cfg.epochs}  Train loss: {total_loss:.2f} Test loss: {test_loss:.2f}"
        )

    torch.save(model, "mnist_ae.pt")

    # collect encodings for analysis
    print("Collecting encodings")
    encodings = []
    with torch.no_grad():
        for data, _ in tqdm(test_loader):
            encodings.append(model.encoder(data.to(device)).cpu())
    encodings = torch.vstack(encodings)
    encodings = np.array(encodings)
    np.save("contrastive_encodings.npy", encodings)

    # decompose into principal components
    tsne = TSNE(n_components=2, verbose=1, random_state=cfg.seed)
    z = tsne.fit_transform(encodings)
    df = pd.DataFrame()
    df["y"] = test_data.test_labels.numpy()
    df["comp-1"] = z[:, 0]
    df["comp-2"] = z[:, 1]

    # plot results
    sns.scatterplot(
        data=df,
        x="comp-1",
        y="comp-2",
        hue=df.y.tolist(),
        palette=sns.color_palette("hls", len(test_data.classes)),
    )
    plt.savefig("contrastive_encodings.pdf")


if __name__ == "__main__":
    train()
