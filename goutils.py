import govars

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


def move_decode(action1d):
    return action1d // govars.SIZE, action1d % govars.SIZE



def action1d_to_onehot(action1d):
    one_hot = np.zeros((govars.ACTION_SPACE))
    one_hot[action1d] = 1

    return one_hot




def one_hot_decode(one_hot):
    return np.argmax(one_hot)



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