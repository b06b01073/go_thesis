from config.vit_config import net_config
from torchvision.models.vision_transformer import VisionTransformer as ViT
import govars
import torch.nn.functional as F

from seed import set_seed
set_seed()



class NNFactory:
    def createModel(self):
        raise NotImplementedError()


class ViTFactory(NNFactory):
    def createModel(self):
        return ViTWrapper()


class Wrapper:
    '''
        An abstract class that serves as a wrapper for actual pytorch neural network.
        The purpose of this interface is to let the derived class add data preprocessing before sending the data to the neural network. 

        The wrapper provide the `forward` and `__call__` method s.t. when interacting with the derived class, the users will feel like they are interacting with the actual pytorch neural network itself.
    '''
    def __init__(self):
        self.net = None


    def forward(self):
        raise NotImplementedError()
    
    
    def __call__(self):
        raise NotImplementedError()
    


    def parameters(self):
        return self.net.parameters()
    

    def to(self, device):
        self.net.to(device)

        return self
    

    def train(self):
        self.net.train()


    def eval(self):
        self.net.eval()



class ViTWrapper(Wrapper):
    def __init__(self):
        self.net = ViT(
            image_size=govars.PADDED_SIZE,
            patch_size=net_config['patch_size'],
            num_classes=net_config['num_class'],
            num_heads=net_config['num_head'],
            num_layers=net_config['encoder_layer'],
            hidden_dim=net_config['embedded_dim'],
            mlp_dim=net_config['embedded_dim'],
            in_channels=govars.FEAT_CHNLS,
            dropout=net_config['dropout']
        )


    def pad_board(self, x):
        '''
            zero padding along the last two dim (height and width of the board)
        '''
        padding = (1, 1, 1, 1)
        return F.pad(x, padding, 'constant', 0) 



    def forward(self, x):
        print('called')
        return self.net.forward(self.pad_board(x))


    def __call__(self, x):
        '''
            Do data preprocessing and send the data to ViT

            Args: 
                x (tensor): the model input
        '''
        return self.net(self.pad_board(x))
    

