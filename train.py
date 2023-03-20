import torch, torch.nn as nn, pickle

from torch.utils.data import DataLoader
from torch.autograd import Variable
from tqdm import tqdm
from model import *


def plot_real_fake(imTensor_real, imTensor_fake, title=''):
    _ , C, H, W = imTensor_real.shape
    grid = make_grid(torch.stack([imTensor_real, imTensor_fake],dim=0).transpose(0,1).reshape(-1, C, H, W)
        , nrow=16, normalize=True).cpu()
    
    fig = plt.figure(figsize=(10,10))
    plt.imshow(grid.permute(1,2,0))
    plt.axis('off')
    plt.title(title)
    plt.show()


class Trainer:

    def __init__(self, dataset_train, dataset_test, config):

        self.config = config
        self.dataloader_train = DataLoader(dataset_train, config['bz'], shuffle=True, num_workers=config['num_workers'])
        self.dataloader_test = DataLoader(dataset_test, config['bz'], shuffle=True, num_workers=config['num_workers'])


        ''' Criterions '''
        self.criterion = nn.BCELoss()
        self.l2_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        ''' Model '''
        self.Generator = generator().to(self.device)
        self.Discriminator = discriminator().to(self.device)

        ''' Optimizers '''
        self.optimD = torch.optim.Adam(self.Generator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))
        self.optimG = torch.optim.Adam(self.Discriminator.parameters(), lr=config['lr'], betas=(config['beta1'], 0.999))


        ''' Statistics '''
        self.train_stats = {
            'losses_d' : [],
            'losses_g' : [],
            'scores_real' : [],
            'scores_fake' : []
        }
        self.test_stats = {

        }
    
    def save_checkpoint(self, epoch):
        torch.save(self.Generator.state_dict, f'./experiment_data/{epoch}_modelG.pt')
        torch.save(self.Discriminator.state_dict, f'./experiment_data/{epoch}_modelD.pt')
        with open(f'./experiment_data/{epoch}_trainstats.pickle', 'wb') as f:
            pickle.dump(self.train_stats, f)
    
    def train(self):

        for epoch in (pbar:=tqdm(range(self.config['epochs']))):
            for batch in self.dataloader_train:
                right_images = batch['right_images']
                right_embed = batch['right_embed']
                wrong_images = batch['wrong_images']

                right_images = Variable(right_images.float()).to(self.device)
                right_embed = Variable(right_embed.float()).to(self.device)
                wrong_images = Variable(wrong_images.float()).to(self.device)

                real_labels = torch.ones(right_images.size(0))
                fake_labels = torch.zeros(right_images.size(0))
                smoothed_real_labels = torch.FloatTensor(Utils.smooth_label(real_labels.numpy(), -0.1))

                real_labels = real_labels.to(self.device)
                fake_labels = fake_labels.to(self.device)
                smoothed_real_labels = smoothed_real_labels.to(self.device)

                # 1. Train discriminator
                self.Discriminator.zero_grad()
                outputs, activation_real = self.Discriminator(right_images, right_embed)
                real_loss = self.criterion(outputs, smoothed_real_labels)       # Why -0.1 for real labels. ?
                real_score = outputs

                outputs, _ = self.Discriminator(wrong_images, right_embed)   # Loss conditioned on CLS
                wrong_loss = self.criterion(outputs, fake_labels)
                wrong_score = outputs

                noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.Generator(right_embed, noise)
                outputs, _ = self.Discriminator(fake_images, right_embed)
                fake_loss = self.criterion(outputs, fake_labels)
                fake_score = outputs
                d_loss = real_loss + fake_loss + wrong_loss       # Loss conditioned on CLS
                d_loss.backward()
                self.optimD.step()

                # 2. Train Generator
                self.Generator.zero_grad()
                noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                noise = noise.view(noise.size(0), 100, 1, 1)
                fake_images = self.Generator(right_embed, noise)
                outputs, activation_fake = self.Discriminator(fake_images, right_embed)
                _, activation_real = self.Discriminator(right_images, right_embed)

                activation_fake = torch.mean(activation_fake, 0)
                activation_real = torch.mean(activation_real, 0)


                ''' Generator loss has the following components
                    1. Regular cross entropy loss
                    2. Feature matching loss (real vs. generated compatibility)
                    3. L1 distance btw. generated & real images
                '''
                g_loss = self.criterion(outputs, real_labels) \
                            + self.config['l2_coef'] * self.l2_loss(activation_fake, activation_real.detach()) \
                            + self.config['l1_coef'] * self.l1_loss(fake_images, right_images)
                g_loss.backward()
                self.optimG.step()

            self.train_stats['losses_d'].append(d_loss.detach().cpu().numpy())
            self.train_stats['losses_g'].append(g_loss.cpu().numpy())
            self.train_stats['scores_real'].append(real_score.detach().cpu().mean().numpy())
            self.train_stats['scores_fake'].append(fake_score.detach().cpu().mean().numpy())
            
            ''' Visual evaluation and checkpoint '''
            stride = self.config['epochs'] // 5
            if stride in [0, epoch-1] or epoch % stride == 0:

                with torch.no_grad():
                    batch = next(iter(self.dataloader_test))
                    right_images = batch['right_images']
                    right_embed = batch['right_embed']
                    txt = batch['txt']

                    right_images = Variable(right_images.float()).to(self.device)
                    right_embed = Variable(right_embed.float()).to(self.device)

                    noise = Variable(torch.randn(right_images.size(0), 100)).to(self.device)
                    noise = noise.view(noise.size(0), 100, 1, 1)
                    fake_images = self.Generator(right_embed, noise)

                    plot_real_fake(right_images, fake_images, f'Test. Epoch {epoch}')
                
                self.save_checkpoint(epoch)
        
        return self.train_stats


