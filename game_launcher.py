import torch
from NNPlayer import NNPlayer
from argparse import ArgumentParser
from Game import Game

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--net', type=str, help='the path to the Pytorch model')
    
    args = parser.parse_args()

    nn_player = NNPlayer(torch.load(args.net))
    game = Game(nn_player)
    game.run()