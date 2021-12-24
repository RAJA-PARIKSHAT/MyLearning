import torch
from torch.utils import data
from torchvision.transforms.transforms import Normalize
from tqdm.auto import tqdm
from torchvision import transforms
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from models import Generator, Critic
from utils import *
import matplotlib.pyplot as plt

n_epochs = 100
z_dim = 64
display_step = 500
learning_rate = 0.0002
batch_size = 128
beta_1 = 0.5
beta_2 = 0.999
c_lambda = 10
crit_repeats = 5
device = 'cpu'


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,),(0.5,))
])


dataloader = DataLoader(MNIST('.', download=True, transform=transform),
    batch_size=batch_size,
    shuffle=True)

generator = Generator(z_dim).to(device)
generator_optimizer = torch.optim.Adam(generator.parameters(), lr = learning_rate, betas= (beta_1, beta_2))

critic = Critic().to(device)
critic_optimizer = torch.optim.Adam(critic.parameters(), lr = learning_rate, betas= (beta_1, beta_2))

generator = generator.apply(weights_init)
critic = critic.apply(weights_init)

cur_step = 0
generator_losses = []
critic_losses = []

for epoch in range(n_epochs):
    
    for real,_ in tqdm(dataloader):
        cur_batch_size = len(real)
        real = real.to(device)
        
        mean_iteration_critic_loss = 0
        
        for _ in range(crit_repeats):
            
            critic_optimizer.zero_grad()
            fake_noise = get_noise(cur_batch_size, z_dim, device= device)
            fake = generator(fake_noise)
            critic_fake_pred = critic(fake.detach())
            critic_real_pred = critic(real)
            
            epsilon = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
            gradient = get_gradient(critic, real, fake.detach(), epsilon)
            gp = gradient_penalty(gradient)
            critic_loss = get_critic_loss(critic_fake_pred, critic_real_pred, gp, c_lambda)
            
            mean_iteration_critic_loss += critic_loss.item() / crit_repeats
            
            critic_loss.backward(retain_graph= True)
            
            critic_optimizer.step()
            
        critic_losses += [mean_iteration_critic_loss]
        
        generator_optimizer.zero_grad()
        fake_noise_2 = get_noise(cur_batch_size, z_dim, device= device)
        
        fake_pred_2 = generator(fake_noise_2)
        critic_pred = critic(fake_pred_2)
        gen_loss = get_gen_loss(critic_pred)
        
        gen_loss.backward()
        generator_optimizer.step()
        
        generator_losses += [gen_loss.item()]
        
        if cur_step % display_step == 0 and cur_step > 0:
            gen_mean = sum(generator_losses[-display_step:]) / display_step
            crit_mean = sum(critic_losses[-display_step:]) / display_step
            print(f"Step {cur_step}: Generator loss: {gen_mean}, critic loss: {crit_mean}")
            show_image_tensor(fake)
            show_image_tensor(real)
            step_bins = 20
            num_examples = (len(generator_losses) // step_bins) * step_bins
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(generator_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Generator Loss"
            )
            plt.plot(
                range(num_examples // step_bins), 
                torch.Tensor(critic_losses[:num_examples]).view(-1, step_bins).mean(1),
                label="Critic Loss"
            )
            plt.legend()
            plt.show()

        cur_step += 1
 
        
        