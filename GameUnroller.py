from GoEnv import Go
import goutils


from seed import set_seed
set_seed()


class GameUnroller:
    def __init__(self):
        self.go_env = Go()

    def unroll(self, game):
        '''
            Args:
                game(list of string): a list of string where each entry represents a move

            Return:
                return a list that stores the features and a list that stores the label
                
        '''

        self.go_env.reset() # reset the game (clear all previous states)

        # this part will be the inputs and labels for the model, where game_moves[i] is the label for game_states[i]
        game_states = [] 
        game_moves = []

        # in SGF the coordinate of a move is represented as [column row] and the origin is at the top-left corner
        for move in game:
            action1d = goutils.move_encode(move) # move is the COLOR[column row], where COLOR is 'B' or 'W'
            game_features = self.go_env.game_features() # get the features for the current board, this will be the input

            game_states.append(game_features)
            game_moves.append(action1d)


            self.go_env.make_move(action1d)

        
        return game_states, game_moves

