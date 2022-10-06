# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "dataroot":'/content/drive/Shared drives/SpotifyMemoria/MM_full/Knn/', #data folder
        "image_size": 64, #all images will be resized to this
        "workers": 2, #num of workers for dataloader
    },
    "train": {
        "batch_size": 128, #batch size during training
        "nc": 3, #Number of channels in the training images
        "nz": 100, # Size of z latent vector (i.e. size of generator input)
        "ngf": 64, # Size of feature maps in generator
        "ndf": 64, # Size of feature maps in discriminator
        #"buffer_size": 1000 #to shuffle images
        "num_epochs": 1000,
        "lr": 0.0002,
        "ngpu": 1,
        #"val_subsplits": 5,
        "optimizer": {
            "type": "adam",
            "beta1": 0.5 # Beta1 hyperparam for Adam optimizers
        },
        "metrics": ["accuracy"],

        
    },
    "model": {
        "input": [128, 128, 3],

        "output": 3
    }
}
