# We can use an image folder dataset the way we have it setup.
# Create the dataset
"""Data Loader"""

import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np


class DataLoader:
    """Data Loader class"""

    @staticmethod
    def load_data(config):
        super().__init__(config)
        dataroot = config.data.dataroot
        dataset = dset.ImageFolder(root=dataroot,
                                transform=transforms.Compose([
                                    transforms.Resize(config.data.image_size),
                                    transforms.CenterCrop(config.data.image_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                ]))
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.train.batch_size,
                                                shuffle=True, num_workers=config.train.workers)

        # Decide which device we want to run on
        device = torch.device("cuda:0" if (torch.cuda.is_available() and config.train.ngpu > 0) else "cpu")

        # Plot some training images
        real_batch = next(iter(dataloader))
        plt.figure(figsize=(8,8))
        plt.axis("off")
        plt.title("Training Images")
        plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))