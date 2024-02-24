import torch
from torch.utils.data import Dataset, DataLoader
from config.dataset_config import dataset_config
from GameUnroller import GameUnroller
import gogame
import goutils
import numpy as np

from seed import set_seed
set_seed()




class GoTrainDataset(Dataset):
    def __init__(self, games):
        """
            Args:
                games (list of string): A list of string, where each entry represents a single game.
        """
        self.games = games
        self.unroller = GameUnroller()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        
        inputs, labels = self.unroller.unroll(game) # we only return the raw feature inputs, things such as zero padding should be done by the user of this dataset
        
        
        # in training set, random symmetric augmentation is performed
        for i in range(len(inputs)):
            inputs[i], labels[i] = gogame.random_symmetry(inputs[i], goutils.action1d_to_onehot(labels[i]))


        return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int_)
    
    
class GoTestDataset(Dataset):
    def __init__(self, games):
        """
            Args:
                games (list of string): A list of string, where each entry represents a single game.
        """
        self.games = games
        self.unroller = GameUnroller()

    def __len__(self):
        return len(self.games)

    def __getitem__(self, idx):
        game = self.games[idx]
        
        inputs, labels = self.unroller.unroll(game) # we only return the raw feature inputs, things such as zero padding should be done by the user of this dataset
        
        return np.array(inputs, dtype=np.float32), np.array(labels, dtype=np.int_)


def get_loader(file, mode):
    '''
        Args: 
            file (str): path to training set
            model (str): use 'train' to get training set, use 'test' to get testing set, training set performs symmetric augmentation during training time
    '''
    assert mode in ['train', 'test']

    with open(file) as f:
        games = f.readlines()
        games = [game.rstrip('\n') for game in games] # strip the newline char
        games = [game.split(',')[1:] for game in games] # extract the moves and discard the file name


    dataset = GoTrainDataset(games) if mode == 'train' else GoTestDataset(games)
    
    return DataLoader(
        dataset, 
        batch_size=dataset_config['batch_size'],
        num_workers=dataset_config['num_workers'],
    )


