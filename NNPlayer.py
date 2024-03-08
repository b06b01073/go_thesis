import torch


class NNPlayer:
    '''
        A class that represents a neural network player
    '''
    def __init__(self, net):
        '''
            Args:
                net(nn.Moduel): a neural network
        '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'model playing on {self.device}')

        self.net = net.to(self.device)
        self.net.eval()


    @torch.no_grad()
    def get_moves_dist(self, x):
        '''
            Get the distribution over moves given the feature
            
            Args:
                x (Torch tensor): a torch tensor that represents the current features of the board

            Returns:
                return the distribution over moves
        '''
        x = x.unsqueeze(dim=0) # turn an input into minibatch with batch size = 1 
        x = self.net(x).squeeze()

        return x