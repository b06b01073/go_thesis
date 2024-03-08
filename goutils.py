import govars

import numpy as np

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

    action2d = move[left_bracket_index + 1: right_bracket_index]
    return action2d_to_action1d(action2d)


def action1d_to_action2d(action1d):
    '''
        convert action in 1d(0 - 361) to 2d coordinate

        Args:
            action1d(int): 0-361

        Returns:
            tuple: (row, col)
            
    '''
    return action1d // govars.SIZE, action1d % govars.SIZE


def readable_action1d(acion1d):
    '''
        convert action in 1d(0 -361) to 2d coordinate in human readable format([a-z a-z])
    '''
    r, c = action1d_to_action2d(acion1d)
    return chr(ord('a') + c) + chr(ord('a') + r)


def action2d_to_action1d(action2d):
    '''
        convert action in 2d to action in 1d

        Args:
            action2d: 2d action in [CR] format 
    '''
    action1d = (ord(action2d[0]) - ord('a')) + (ord(action2d[1]) - ord('a')) * 19 # action1d = col + row * 19, so that action2d = (row, col), see gogame.py(next_state function)

    return action1d


def action1d_to_onehot(action1d):
    one_hot = np.zeros((govars.ACTION_SPACE))
    one_hot[action1d] = 1

    return one_hot

