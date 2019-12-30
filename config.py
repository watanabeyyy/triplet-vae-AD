import numpy as np
import torch
import random


class config():
    # model setup
    z_dim = 2

    # paths
    dataset_path = "/images/"
    target = "/tar/"
    number = 0
    weights_path = "triplet_weights"

    # cuda setup
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if cuda else {}

    # set seed
    np.random.seed(2019)
    random.seed(2019)
    torch.manual_seed(2019)
    if cuda:
        torch.cuda.manual_seed_all(2019)


config = config()
