import numpy as np
import gogame
import govars
import goutils

from seed import set_seed
set_seed()


class Go:
    def __init__(self):
        self.reset()


    def reset(self):
        '''
            reset all the states to initial states
        '''
        self.game_state = gogame.init_state(size=govars.SIZE)
        self.recent_moves = np.zeros((govars.LAST_MOVE_PLANES, govars.SIZE, govars.SIZE))
        self.empty_plane = np.ones((govars.SIZE, govars.SIZE))
        self.invalid_move = np.zeros((govars.SIZE, govars.SIZE))
        self.one_plane = np.zeros((govars.SIZE, govars.SIZE))
        self.capture_size = np.zeros((govars.CAPTURE_PLANES, govars.SIZE, govars.SIZE))
        self.self_atari_size = np.zeros((govars.SELF_ATARI_PLANES, govars.SIZE, govars.SIZE)) 
        self.liberty = np.zeros((govars.LIBERTY_PLANES, govars.SIZE, govars.SIZE))
        self.last_move = None
    

    def is_ended(self):
        return np.max(self.game_state[govars.DONE_CHNL]) == 1


    def build_recent_moves(self, action1d):
        last_move = np.zeros((govars.SIZE, govars.SIZE))
        if action1d != govars.PASS:
            action2d = goutils.action1d_to_action2d(action1d)
            last_move[action2d] = 1
        self.recent_moves[:-1] = self.recent_moves[1:] 
        self.recent_moves[-1] = last_move
        self.last_move = last_move


    def build_capture_size(self):
        turn = int(np.max(self.game_state[govars.TURN_CHNL]))
        opponent_channel = govars.WHITE if (turn == govars.BLACK) else govars.BLACK
        self.capture_size = np.zeros((govars.CAPTURE_PLANES, govars.SIZE, govars.SIZE))
        valid_actions = np.argwhere(self.game_state[govars.INVD_CHNL] == 0)
        original_pieces = np.count_nonzero(self.game_state[opponent_channel])


        for move in valid_actions:
            move1d = move[0] * govars.SIZE + move[1]
            new_pieces = np.count_nonzero(gogame.next_state(self.game_state, move1d)[opponent_channel])
            captures = original_pieces - new_pieces

            if captures == 0:
                continue
            elif captures < govars.CAPTURE_PLANES:
                self.capture_size[captures - 1, move[0], move[1]] = 1
            else:
                self.capture_size[govars.CAPTURE_PLANES - 1, move[0], move[1]] = 1

    def build_self_atari_size(self):
        # pass and pretend it's oppenent's turn
        self.self_atari_size = np.zeros((govars.SELF_ATARI_PLANES, govars.SIZE, govars.SIZE))
        game_state = gogame.next_state(self.game_state, govars.PASS, enable_ended=False) # disable game ends, since this pass is fake
        turn = int(np.max(game_state[govars.TURN_CHNL]))
        opponent_channel = govars.WHITE if (turn == govars.BLACK) else govars.BLACK

        valid_actions = np.argwhere(game_state[govars.INVD_CHNL] == 0)
        original_pieces = np.count_nonzero(game_state[opponent_channel])
    

        for move in valid_actions:

            move1d = move[0] * govars.SIZE + move[1]
            new_pieces = np.count_nonzero(gogame.next_state(game_state, move1d)[opponent_channel])
            captures = original_pieces - new_pieces
            if captures == 0:
                continue
            elif captures < govars.SELF_ATARI_PLANES:
                self.self_atari_size[captures - 1, move[0], move[1]] = 1
            else:
                self.self_atari_size[govars.SELF_ATARI_PLANES - 1, move[0], move[1]] = 1


        


    def build_game_feature(self, action1d):
        game_state = np.copy(self.game_state)

        # build the plane of empty
        black = game_state[govars.BLACK].astype(np.int32)
        white = game_state[govars.WHITE].astype(np.int32)
        self.empty_plane = 1 - np.bitwise_or(black, white)


        self.build_recent_moves(action1d) # build the planes of recent moves
        
        
        # the below features are computational epensive
        # self.build_liberty()

        # self.build_capture_size() # build the capture size

        # self.build_self_atari_size() # build the self atari size




    def make_move(self, action1d):
        self.game_state = gogame.next_state(self.game_state, action1d)


        self.build_game_feature(action1d)

        return np.copy(self.game_state)
    
    def render(self):
        return print(gogame.str(self.game_state))
    

    def game_features(self):
        '''Do feature extraction here
        '''
        game_feature = np.stack((
            self.game_state[govars.BLACK],
            self.game_state[govars.WHITE],
            self.game_state[govars.TURN_CHNL],
            self.game_state[govars.INVD_CHNL],
            self.empty_plane,
            self.one_plane,
        ), axis=0)

        game_feature = np.concatenate((
            game_feature, 
            self.recent_moves,
            # self.capture_size,
            # self.self_atari_size,
        ), axis=0)


        return np.copy(game_feature).astype(np.float32)
    

    def is_valid(self, action1d):
        row, col = goutils.action1d_to_action2d(action1d)
        if self.game_state[govars.INVD_CHNL, row, col]:
            return False
    
        return True


    def get_state(self):
        return np.copy(self.game_state)
    
