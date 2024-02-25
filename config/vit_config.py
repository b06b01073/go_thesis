# this file stores global data such as hyperparameters for the ViT, and also the training parameters for ViT


# the net_config dictionary save the data related to the NN
net_config = {
    'patch_size': 7,
    'embedded_dim': 768,
    'encoder_layer': 12,
    'num_head': 8, 
    'dropout': 0.05,
    'num_class': 361, # number of moves(19 * 19)
}


# hyperparams for optimizer
optim_config = {
    'lr': 1e-4, 
    'weight_decay': 0,
}


