import torch
from torch.autograd import grad
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
import torch.nn as nn

def show_image_tensor(image_tensors, num_of_images = 25, size = (1,28,28)):
    
    image_tensors = (image_tensors + 1)/2
    image_tensors = image_tensors.detach().cpu()
    image_grid = make_grid(image_tensors[:num_of_images], nrow = 5)
    plt.imshow(image_grid.permute(1,2,0).squeeze())
    
def make_grad_hook():
    '''
    Function to keep track of gradients for visualization purposes, 
    which fills the grads list when using model.apply(grad_hook).
    '''
    grads = []
    def grad_hook(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            grads.append(m.weight.grad)
    return grads, grad_hook

def get_noise(num_samples, z_dim, device = 'cpu'):
    return torch.randn(num_samples, z_dim, device = device)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    if isinstance(m, nn.BatchNorm2d):
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
        torch.nn.init.constant_(m.bias, 0)
        

def get_gradient(critic, real, fake, epsilon):
    
    sample = epsilon*real + (1-epsilon)*fake
    
    scores = critic(sample)
    gradients = torch.autograd.grad(
        inputs= sample,
        outputs= scores,
        grad_outputs= torch.ones_like(scores),
        create_graph= True,
        retain_graph= True
    )[0]
    
    return gradients

def gradient_penalty(gradient):
    
    gradient = gradient.view(len(gradient), -1)
    
    gradient_norm = gradient.norm(2, dim =1)
    penalty = torch.mean((gradient_norm - 1)**2)
    
    return penalty

def get_gen_loss(critic_fake_pred):
    
    loss = -1 * torch.mean(critic_fake_pred)
    
    return loss


def get_critic_loss(critic_fake_pred, critic_real_pred, gp, c_lamba):
    
    critic_loss = -torch.mean(critic_real_pred) + torch.mean(critic_fake_pred) + c_lamba*gp
    
    return critic_loss


    
    
    