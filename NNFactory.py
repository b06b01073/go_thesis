from config.vit_config import net_config
from torchvision.models.vision_transformer import VisionTransformer as ViT
import govars
import torch.nn.functional as F
import torch.nn as nn
import torch
from einops import repeat


from seed import set_seed
set_seed()


class NNFactory:
    def createModel(self):
        raise NotImplementedError()


class ViTFactory(NNFactory):
    def createModel(self):
        return ViTWrapper()
    


class Nature2016Factory(NNFactory):
    def createModel(self):
        return Nature2016Net()



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
    
    
    def __call__(self, x):
        return self.forward(x)
    


    def parameters(self):
        return self.net.parameters()
    

    def to(self, device):
        self.net.to(device)

        return self
    

    def train(self):
        self.net.train()


    def eval(self):
        self.net.eval()


    def state_dict(self):
        return self.net.state_dict()



class ViTWrapper(Wrapper):
    '''
        THe ViT requires a wrapper since ViT is an api provided by Pytorch, we have no control of how the data is processed. Therefore, an wrapper is required to perform data preprocessing
    '''
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
        '''
            Do a forward call
        '''

        return self.net.forward(self.pad_board(x))





class Nature2016Net(nn.Module):
    def __init__(self, hidden_layers=6, hidden_dim=64):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=govars.FEAT_CHNLS, out_channels=hidden_dim, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
        )


        # create the hidden layers with `hiden_layers` layers
        self.hidden_layers = nn.ModuleList([])
        for _ in range(hidden_layers):
            self.hidden_layers.extend([
                nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1, padding=1), 
                nn.ReLU()
            ])
            

        self.output_layer = nn.Sequential(
            PositionalBiasConv2d(in_channels=hidden_dim, out_channels=1, kernel_size=1, stride=1),
            nn.Flatten(), # (B, 19 * 19)
        )



    def forward(self, x):
        '''
            Args:
                x (tenser): the model input

            Returns:
                The `log likelihood` of moves
        '''
        x = self.input_layer(x)
        for layer in self.hidden_layers:
            x = layer(x)

        x = self.output_layer(x)
        return x



class PositionalBiasConv2d(nn.Module):
    '''
        quote from the AlphaGo paper:
            "The final layer convolves 1 filter of kernel size 1 x 1 with stride 1, with a different bias for each position"

        This class applies the nn.Conv2d without bias, then it add a (1, 19, 19) learnable parameter (`nn.Parameter`) whichs serve as "different bias for each position". 
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()

        self.conv2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        
        self.bias = nn.Parameter((torch.rand(1, govars.SIZE, govars.SIZE) - 0.5) * 0.1) # initialized in the range [-0.5, 0.5)


    def forward(self, x):
        '''
            Args:
                x (tensor): the input of model, x is of the shape (B C 19 19)
        '''

        x = self.conv2d(x) # x: (B, 1, 19, 19)
        
        batch_size, _, _, _ = x.shape
        bias = repeat(self.bias, 'c h w -> b c h w', b=batch_size) # repeats the self.bais along the batch dimension, bias is of the shape (B, 1, 19, 19)

        x = x + bias

        return x
        
