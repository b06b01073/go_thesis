from GoEnv import Go
import os
import goutils
import torch

class Game:
    def __init__(self, nn_player):
        '''
            Args:
                player (NNPlayer): a player instance, which is a neural network
        '''

        self.go_env = Go()
        self.nn_player = nn_player
        self.human_player = None # 0 if black, 1 if white
        self.cur_player = None # 0 if black, 1 if white


    def init_game(self):
        '''
            Start a game by letting the user choose the desired color to start with
        '''
        color = input('Color to start with [B|W]:')
        if color == 'B' or color == 'b':
            self.human_player = 0
        elif color == 'W' or color == 'w':
            self.human_player = 1
        else:
            print('No such option')
            exit()
            
        self.cur_player = 0
        self.nn_last_move = None
        self.human_last_move = None
        self.go_env.reset()


    def run(self):
        '''
            Entry point of the Game instance
        '''
        self.init_game()
        self.play()

    
    def human_move(self):
        '''
            Read a move from human until the move is legal
        '''
        while True:
            action2d = input('Your move: ')
            action1d = goutils.action2d_to_action1d(action2d)

            if self.go_env.is_valid(action1d):
                return action1d
            else:
                print("Bruh, that's not nice")


    def play(self):
        '''
            get the moves from human and NNPlayer
        '''
        while not self.go_env.is_ended():
            if self.cur_player == self.human_player:
                os.system('clear') # use 'os.system('cls') on Windows
                self.go_env.render()
                print('All the moves are represented in [Col Row] format.')
                print(f'Your last move is {self.human_last_move}, opponent\'s last move is {self.nn_last_move}')

                move = self.human_move()
                self.human_last_move = goutils.readable_action1d(move)

                self.go_env.make_move(move)
                
            else:
                features = torch.from_numpy(self.go_env.game_features())
                features = features.to(self.nn_player.device)

                move_dist = self.nn_player.get_moves_dist(features)
                for i in range(len(move_dist)):
                    if not self.go_env.is_valid(i):
                        move_dist[i] = float('-inf')

                move_dist = torch.nn.functional.softmax(move_dist, dim=0)
                move = torch.argmax(move_dist).item()

                self.nn_last_move = goutils.readable_action1d(move)
                
                self.go_env.make_move(move)


            self.cur_player = (self.cur_player + 1) % 2