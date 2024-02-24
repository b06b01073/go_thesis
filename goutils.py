import govars
from GoEnv import Go


import random
import numpy as np

import torch

from seed import set_seed
set_seed()



def move_encode(move):
    '''
        Args:
        move (string): a string of the form COLOR[CR], where COLOR is 'B' or 'W', C is the column, R is the row. If CR are empty, then it represents a pass move
    '''

    left_bracket_index = move.index('[')
    right_bracket_index = move.index(']')

    # CR is empty
    if left_bracket_index + 1 == right_bracket_index:
        return govars.PASS

    coord = move[left_bracket_index + 1: right_bracket_index]
    move_1d = (ord(coord[0]) - ord('a')) + (ord(coord[1]) - ord('a')) * 19 # action1d = col + row * 19, so that action2d = (row, col), see gogame.py(next_state function)
    return move_1d


def move_decode_char(action1d):
    # decode the 1d move to the 2d char coord
    action2d = action1d // govars.SIZE, action1d % govars.SIZE
    return chr(ord('a') + action2d[0]) + chr(ord('a') + action2d[1])

def move_decode(action1d):
    return action1d // govars.SIZE, action1d % govars.SIZE



def action1d_to_onehot(action1d):
    one_hot = np.zeros((govars.ACTION_SPACE))
    one_hot[action1d] = 1

    return one_hot



def pad_board(state):
    # zero padding
    return np.pad(state, ((0, 0), (1, 1), (1, 1)), mode='constant', constant_values=0)



def move_2d_encode(move_1d, state_size=govars.SIZE):

    coord_0 = move_1d % state_size
    coord_1 = move_1d // state_size

    return (coord_0, coord_1)


def one_hot_decode(one_hot):
    return np.argmax(one_hot)



def board_augment(state, move):
    # note that the first channel of state is the feature channel, so we need to flip along the second channel, and rotate along the (1, 2) channels
    state_size = state.shape[1]
    
    move = one_hot_decode(move)

    move_2d = move_2d_encode(move, state_size)
    move_1d = np.zeros((govars.ACTION_SPACE,))

    # move_board = np.zeros((govars.SIZE, govars.SIZE))
    move_board = np.zeros((state_size, state_size))
    if move != govars.PASS:
        move_board[move_2d[1], move_2d[0]] = 1
        move_1d[-1] = 1


    # flip the board with 0.5 prob
    flip = random.random() > 0.5 # 0.5 to filp
    if flip:
        state = np.flip(state, 2)
        move_board = np.flip(move_board, 1)


    # rotate the board
    rotate_times = random.randint(a=0, b=3) # counterclockwise rotate 90 * rotate_deg deg
    state = np.rot90(state, rotate_times, axes=(1, 2))
    move_board = np.rot90(move_board, rotate_times, axes=(0, 1))

    move_1d[:-1] = move_board.flatten()
    return state, np.argmax(move_1d)


def test_time_predict(board, net, device):
    preds = torch.zeros((govars.ACTION_SPACE-1,)).to(device)
    rotate_k = [0, 1, 2, 3] # rotate degree
    flip = [False, True]

    augments = [(k, f) for k in rotate_k for f in flip]

    for (rotate_times, f) in augments:
        # augmentation and prediction
        augmented_board = board
        if f:
            augmented_board = torch.flip(augmented_board, dims=(2,))
        augmented_board = torch.rot90(augmented_board, k=rotate_times, dims=(1, 2))

        augmented_board = augmented_board.unsqueeze(dim=0)
        augmented_preds = net(augmented_board).squeeze()[:-1] # discard the PASS move

        # mask invalid move
        for i in range(govars.ACTION_SPACE - 1):
            action2d = move_decode(i)
            if augmented_board[0, govars.INVD_CHNL, action2d[0] + 1, action2d[1] + 1]:
                augmented_preds[i] = float('-inf')

        augmented_preds = torch.softmax(augmented_preds, dim=0).view(govars.SIZE, govars.SIZE)

        # restore the prediction to the original coord system
        # note that it "have" to be done in the reverse order
        augmented_preds = torch.rot90(augmented_preds,k=-rotate_times, dims=(0, 1))
        if f:
            augmented_preds = torch.flip(augmented_preds, dims=(1,))

        augmented_preds = augmented_preds.flatten()
        preds += augmented_preds
    return preds / 8


        

def mask_moves(pred):
    pred[govars.PASS] = float('-inf') # mask the pass move
    return pred