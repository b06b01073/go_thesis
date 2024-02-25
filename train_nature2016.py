from argparse import ArgumentParser
import torch.nn as nn


from NNFactory import Nature2016Factory
from OptimFactory import SGDFactory
from Trainer import Trainer
from config.nature2016_config import optim_config
import GoDataset
from seed import set_seed


import os


if __name__ == '__main__':
    set_seed()

    parser = ArgumentParser()
    # parser.add_argument('--config_path', type=str, default='./hyperparams.json', help='path to the hyperparams file')
    parser.add_argument('--save_path', type=str, default='./trained_model')
    parser.add_argument('--file_name', type=str, default='Nature2016.pth', help='name of the saved model')
    parser.add_argument('--train', type=str, default='./dataset/train.txt', help='path to the train set')
    parser.add_argument('--test', type=str, default='./dataset/test.txt', help='path to the test set')

    args = parser.parse_args()
    

    # crreate the folder to save the model if the path does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    

    net = Nature2016Factory().createModel()
    optim = SGDFactory().create_optim(net, optim_config)
    loss_func = nn.CrossEntropyLoss()


    trainer = Trainer(
        net, 
        optim, 
        loss_func, 
        os.path.join(args.save_path, args.file_name),
    )


    train_set = GoDataset.get_loader(args.train, 'train')
    test_set = GoDataset.get_loader(args.test, 'test')


    trainer.fit(
        train_set,
        test_set,
        'nature2016.log',
        'nature2016_latest.pth'
    )