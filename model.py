# DCGAN for mnist dataset

import torch
import torch.nn as nn
import torch.nn.functional as F





# generator 
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # project the latent z of dimension 1x100 to a vector of dimension 7x7x512
        self.fc = nn.Linear(in_features=100, out_features=7*7*512)
        self.bn_fc = nn.BatchNorm1d(7*7*512)
        
        # upsampling 
        self.up1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(256)
        
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) # 14x14 -> 28x28
        self.bn2 = nn.BatchNorm2d(128)
        
        # refining
        self.refine = nn.ConvTranspose2d(128, 128, 3, 1, 1) # 28x28 -> 28x28
        self.bn3 = nn.BatchNorm2d(128)
        
        # output
        self.out = nn.Conv2d(128, 1, 3, 1, 1)
        
        

    def forward(self, z):
        # latent projection and reshaping
        z = self.fc(z)
        z = self.bn_fc(z)
        z = F.relu(z)
        z = z.view(-1, 512, 7, 7)

        # upsampling
        z = self.up1(z)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.up2(z)
        z = self.bn2(z)
        z = F.relu(z)
        
        # refine
        z = self.refine(z)
        z = self.bn3(z)
        z = F.relu(z)
        
        # out
        z = self.out(z)
        
        return torch.tanh(z)
    
    
    
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=5, padding=2, stride=2)
        self.down2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.out = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=7)

        
    def forward(self, x):
        x = self.down1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.down2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.out(x)
        # x = F.leaky_relu(x, 0.2)
        return x
    
    



# generator 
class CGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        
        # project the latent z of dimension 1x100 to a vector of dimension 7x7x512
        self.fc = nn.Linear(in_features=100+10, out_features=7*7*512)
        self.bn_fc = nn.BatchNorm1d(7*7*512)
        
        # upsampling 
        self.up1 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=4, stride=2, padding=1) # 7x7 -> 14x14
        self.bn1 = nn.BatchNorm2d(256)
        
        self.up2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1) # 14x14 -> 28x28
        self.bn2 = nn.BatchNorm2d(128)
        
        # refining
        self.refine = nn.ConvTranspose2d(128, 128, 3, 1, 1) # 28x28 -> 28x28
        self.bn3 = nn.BatchNorm2d(128)
        
        # output
        self.out = nn.Conv2d(128, 1, 3, 1, 1)
        
        

    def forward(self, z, c):
        # latent projection and reshaping
        
        z = torch.cat((z, c), dim=1)
        
        z = self.fc(z)
        z = self.bn_fc(z)
        z = F.relu(z)
        z = z.view(-1, 512, 7, 7)

        # upsampling
        z = self.up1(z)
        z = self.bn1(z)
        z = F.relu(z)
        z = self.up2(z)
        z = self.bn2(z)
        z = F.relu(z)
        
        # refine
        z = self.refine(z)
        z = self.bn3(z)
        z = F.relu(z)
        
        # out
        z = self.out(z)
        
        return torch.tanh(z)



class CDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(in_channels=1+10, out_channels=256, kernel_size=5, padding=2, stride=2)
        self.down2 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=5, padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(512)
        self.out = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=7)

        
    def forward(self, x):
        x = self.down1(x)
        x = F.leaky_relu(x, 0.2)
        x = self.down2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        
        x = self.out(x)
        # x = F.leaky_relu(x, 0.2)
        return x


        

if __name__=="__main__":
    z = torch.rand((5,100))
    generator = Generator()
    discriminator = Discriminator()
    generator.eval()
    discriminator.eval()
    g = generator(z)
    print(g.shape)
    d = discriminator(g)
    print(d.shape)
        
