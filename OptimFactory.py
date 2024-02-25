import torch.optim as optim
from seed import set_seed
set_seed()


class OptimFactory:
    def create_optim(self, net, optim_config):
        '''
            Args:
                net(nn.Module): neural network 
                optim_config (dict): dictionary that stores the hyperparams of optimizer
        '''
        raise NotImplementedError()
    

class AdamFactory(OptimFactory):
    def create_optim(self, net, optim_config):
        return optim.Adam(net.parameters(), lr=optim_config['lr'], weight_decay=optim_config['weight_decay'])
    

class SGDFactory(OptimFactory):
    def create_optim(self, net, optim_config):
        return optim.SGD(net.parameters(), lr=optim_config['lr'])