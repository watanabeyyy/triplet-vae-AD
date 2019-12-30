import torch
from torch import optim
import numpy as np
from torchvision.utils import save_image
import os
from matplotlib import pyplot as plt
from torch.nn import BCELoss, TripletMarginLoss
import time, copy

from config import config
from vae import vae_model

os.makedirs("results", exist_ok=True)
os.makedirs("model_weights", exist_ok=True)


def recon_loss(recon_x, x):
    bce = BCELoss(reduction="sum")(recon_x + 1e-10, x + 1e-10) / x.shape[0]
    return bce


def kld_loss(mu, logvar):
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / logvar.shape[0]
    return kld


def triplet_loss(anchor, positive, negative):
    # return torch.sum(torch.sum((anchor - positive).pow(2), 1) - torch.sum((anchor - negative).pow(2), 1))
    # anchor = normalize(anchor)
    # positive = normalize(positive)
    # negative = normalize(negative)
    anchor = anchor.view(anchor.shape[0], -1)
    positive = positive.view(positive.shape[0], -1)
    negative = negative.view(negative.shape[0], -1)
    return TripletMarginLoss(reduction="sum")(anchor, positive, negative) / anchor.shape[0]


def train_model(model, optimizer, scheduler, num_epochs, dataloaders):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000
    best_epoch = 0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            BCE = 0
            KLD = 0
            TRIPLET = 0

            # Iterate over data.
            for batch_idx, (anchor, positive, negative) in enumerate(dataloaders[phase]):
                anchor = anchor.to(config.device)
                positive = positive.to(config.device)
                negative = negative.to(config.device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    recon_a, mu_a, logvar_a, latents_a = model(anchor, False)
                    recon_p, mu_p, logvar_p, latents_p = model(positive, False)
                    recon_n, mu_n, logvar_n, latents_n = model(negative, False)
                    loss1 = recon_loss(recon_a, anchor) + recon_loss(recon_p, positive)
                    loss2 = kld_loss(mu_a, logvar_a) + kld_loss(mu_p, logvar_p)
                    loss3 = triplet_loss(latents_a, latents_p, latents_n)
                    loss = loss1 + 100 * loss2 + 500 * loss3

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                    else:
                        if batch_idx == 0:
                            n = min(anchor.shape[0], 4)
                            idxs = np.random.randint(0, anchor.shape[0], n)
                            comparison = torch.cat([positive[idxs], recon_p[idxs]])
                            save_image(comparison.cpu(),
                                       './results/p_' + str(epoch) + '.png', nrow=n)
                            n = min(negative.shape[0], 4)
                            idxs = np.random.randint(0, negative.shape[0], n)
                            comparison = torch.cat([negative[idxs], recon_n[idxs]])
                            save_image(comparison.cpu(),
                                       './results/n_' + str(epoch) + '.png', nrow=n)

                # statistics
                running_loss += loss.item()
                BCE += loss1.item()
                KLD += loss2.item()
                TRIPLET += loss3.item()

            epoch_loss = running_loss / (batch_idx + 1)
            BCE = BCE / (batch_idx + 1)
            KLD = KLD / (batch_idx + 1)
            TRIPLET = TRIPLET / (batch_idx + 1)

            # print('{} Loss: {:.4f} Acc: {:.4f}'.format(
            #     phase, epoch_loss))
            print('{} BCE: {:.4f}, KLD: {:.4f}, TRIPLET: {:.4f}'.format(phase, BCE, KLD, TRIPLET))

            # deep copy the model
            if phase == 'val' and TRIPLET < best_loss:
                best_epoch = epoch
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best epoch:{}, val Loss: {:4f}'.format(best_epoch, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


if __name__ == "__main__":

    from torchvision import transforms
    from torch.utils.data import DataLoader
    from utils import MDDataset
    from sklearn.model_selection import train_test_split

    dataset_path = config.dataset_path
    target = config.target
    number = config.number
    weights_path = config.weights_path

    global cuda
    global device
    global model
    global optimizer

    cuda = config.cuda
    device = config.device
    model = vae_model.BetaVAE_B(z_dim=config.z_dim, nc=1).to(device)

    model_dict = model.state_dict()
    for i, (name, param) in enumerate(model.named_parameters()):
        param.requires_grad = True
        print(i, name)
    # model.load_state_dict((torch.load("model_weights/freeze_weights", map_location="cuda" if cuda else "cpu")))

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 27])

    num_epochs = 30

    all_data = np.load("dataset/data.npy")
    train_data, val_data = train_test_split(all_data, test_size=0.2)
    print(train_data.shape, val_data.shape)
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.5],std=[0.225])
    ])
    train_dataset = MDDataset(train_data, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_dataset = MDDataset(val_data, transform=transforms.ToTensor())
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False, num_workers=4)
    dataloaders = {"train": train_loader, "val": val_loader}

    model = train_model(model, optimizer, scheduler, num_epochs, dataloaders)
    torch.save(model.state_dict(), 'model_weights/weights')
