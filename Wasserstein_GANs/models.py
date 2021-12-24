import torch


class Generator(torch.nn.Module):
    
    
    
    def __init__(self, z_dim = 10, im_channel = 1, hidden_dim = 64):
        super().__init__()
        self.z_dim = z_dim
        self.generator = torch.nn.Sequential(
            self.make_generator_block(self.z_dim, hidden_dim*4),
            self.make_generator_block(4*hidden_dim, 2*hidden_dim, kernel_size= 4, stride=1),
            self.make_generator_block(2*hidden_dim, hidden_dim),
            self.make_generator_block(hidden_dim, im_channel, kernel_size=4, final_layer= True)
            
        )
    
    def make_generator_block(self, input_channels, output_channels, kernel_size = 3, stride = 2, final_layer = False):
        
        if not final_layer:
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels= input_channels,
                                         out_channels= output_channels,
                                         kernel_size= kernel_size,
                                         stride= stride),
                torch.nn.BatchNorm2d(output_channels),
                torch.nn.ReLU(inplace= True)
            )
        else:
            return  torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels= input_channels,
                                         out_channels= output_channels,
                                         kernel_size= kernel_size,
                                         stride= stride),
                torch.nn.Tanh()
            )
    
    def forward(self, noise):
        
        x = noise.view(len(noise), self.z_dim, 1,1)
        return self.generator(x)
    
    
class Critic(torch.nn.Module):
    
    def __init__(self, image_channel = 1, hidden_dim = 64):
        
        super().__init__()
        
        self.hidden_dim = 64
        self.critic = torch.nn.Sequential(
            self.make_critic_block(image_channel, hidden_dim),
            self.make_critic_block(hidden_dim, hidden_dim * 2),
            self.make_critic_block(hidden_dim * 2, 1, final_layer=True),
        )
        
    def make_critic_block(self, image_channel, output_channel, kernel_size = 4, stride = 2, final_layer = False ):
        
        if not final_layer:
            
            return torch.nn.Sequential(
                torch.nn.Conv2d(image_channel, output_channel, kernel_size= kernel_size, stride= stride),
                torch.nn.BatchNorm2d(output_channel),
                torch.nn.LeakyReLU(0.2, inplace= True)   
            )
            
        else:
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels= image_channel, out_channels= output_channel, kernel_size= kernel_size, stride= stride)
            )
            
    def forward(self, image):
        
        result = self.critic(image)
        
        return result.view(len(result), -1)