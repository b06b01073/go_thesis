from argparse import ArgumentParser
import torch.nn as nn

from NNFactory import ViTFactory
from OptimFactory import AdamFactory
from Trainer import Trainer
from config.vit_config import *
from config.training_config import *
import GoDataset
from seed import set_seed
from TrainingChock import settings_checker, path_checker


import os


if __name__ == '__main__':
    set_seed()

    parser = ArgumentParser()
    # parser.add_argument('--config_path', type=str, default='./hyperparams.json', help='path to the hyperparams file')
    parser.add_argument('--train', type=str, default='./dataset/train.txt', help='path to the train set')
    parser.add_argument('--test', type=str, default='./dataset/test.txt', help='path to the test set')

    parser.add_argument('--save_path', type=str, default='./trained_model')


    # we want the users to explicitly type out the path
    parser.add_argument('--file_name', type=str, required=True, help='name of the saved model (best performing model)')
    parser.add_argument('--log_path', type=str, required=True, help='file name of the log (accuracy during training)')
    parser.add_argument('--latest_path', type=str, required=True, help='file name of the latest iteration of the model')

    args = parser.parse_args()
    

    # crreate the folder to save the model if the path does not exist
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)



    path_checker(
        args.log_path,
        args.latest_path,
        os.path.join(args.save_path, args.file_name)
    )


    settings_checker(
        args, 
        optim_config, 
        net_config, 
        training_config
    )

        
    # the training process starts here
    net = ViTFactory().createModel()
    optim = AdamFactory().create_optim(net, optim_config)
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
        args.log_path,
        args.latest_path
    )