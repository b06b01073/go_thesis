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

    coord = move[left_bracket_index + 1: right_bracket_index]
    move_1d = (ord(coord[0]) - ord('a')) + (ord(coord[1]) - ord('a')) * 19 # action1d = col + row * 19, so that action2d = (row, col), see gogame.py(next_state function)
    return move_1d


def action1d_to_action2d(action1d):
    '''
        convert action in 1d(0 - 361) to 2d coordinate
        Args:
    '''
    return action1d // govars.SIZE, action1d % govars.SIZE



def action1d_to_onehot(action1d):
    one_hot = np.zeros((govars.ACTION_SPACE))
    one_hot[action1d] = 1

    return one_hot

