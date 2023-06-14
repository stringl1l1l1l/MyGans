import argparse
import os
import numpy as np
import math
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
from collections import OrderedDict
from data.cup_plus import Cup

os.makedirs("images/wgan", exist_ok=True)
os.makedirs("models/wgan", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=3000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr_g", type=float, default=0.001, help="generator learning rate")
parser.add_argument("--lr_d", type=float, default=0.001, help="discriminator learning rate")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
parser.add_argument("--clip_value", type=float, default=10000, help="lower and upper clip value for disc. weights")
parser.add_argument("--sample_interval", type=int, default=1000, help="interval between image samples")
parser.add_argument("--model_interval", type=int, default=10000, help="interval between model samples")
parser.add_argument("--model_depth", type=int, default=4, help="depth of model's net")
parser.add_argument("--gp_weight", type=int, default=10, help="gradient penalty weight")
parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient")

opt = parser.parse_args()

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class Generator(nn.Module):
    def __init__(self,
                 dimLatentVector = opt.latent_dim,
                 dimOutput = opt.channels,
                 dimModelG = opt.img_size,
                 depthModel = opt.model_depth,
                 generationActivation=nn.Tanh()):
        super(Generator, self).__init__()

        self.depthModel = depthModel
        self.refDim = dimModelG

        self.initFormatLayer(dimLatentVector)

        currDepth = int(dimModelG * (2**depthModel))

        sequence = OrderedDict([])
        # input is Z, going into a convolution
        sequence["batchNorm0"] = nn.BatchNorm2d(currDepth)
        sequence["relu0"] = nn.ReLU(True)

        for i in range(depthModel):

            nextDepth = int(currDepth / 2)

            # state size. (currDepth) x 2**(i+1) x 2**(i+1)
            sequence["convTranspose" + str(i+1)] = nn.ConvTranspose2d(
                currDepth, nextDepth, 4, 2, 1, bias=False)
            sequence["batchNorm" + str(i+1)] = nn.BatchNorm2d(nextDepth)
            sequence["relu" + str(i+1)] = nn.ReLU(True)

            currDepth = nextDepth

        sequence["outlayer"] = nn.ConvTranspose2d(
            dimModelG, dimOutput, 4, 2, 1, bias=False)
        
        self.outputAcctivation = generationActivation

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

    def initFormatLayer(self, dimLatentVector):

        currDepth = int(self.refDim * (2**self.depthModel))
        self.formatLayer = nn.ConvTranspose2d(
            dimLatentVector, currDepth, 4, 1, 0, bias=False)

    def forward(self, input):
        x = input.view(-1, input.size(1), 1, 1)
        x = self.formatLayer(x)
        x = self.main(x)
        
        if self.outputAcctivation is None:
            return x
        return self.outputAcctivation(x)


class Discriminator(nn.Module):
    def __init__(self,
                 dimInput = opt.channels,
                 dimModelD = opt.img_size,
                 sizeDecisionLayer = opt.channels,
                 depthModel = opt.model_depth):
        super(Discriminator, self).__init__()

        currDepth = dimModelD
        sequence = OrderedDict([])

        # input is (nc) x 2**(depthModel + 3) x 2**(depthModel + 3)
        sequence["convTranspose" +
                 str(depthModel)] = nn.Conv2d(dimInput, currDepth,
                                              4, 2, 1, bias=False)
        sequence["relu" + str(depthModel)] = nn.LeakyReLU(0.2, inplace=True)

        for i in range(depthModel):

            index = depthModel - i - 1
            nextDepth = currDepth * 2

            # state size.
            # (currDepth) x 2**(depthModel + 2 -i) x 2**(depthModel + 2 -i)
            sequence["convTranspose" +
                     str(index)] = nn.Conv2d(currDepth, nextDepth,
                                             4, 2, 1, bias=False)
            sequence["batchNorm" + str(index)] = nn.BatchNorm2d(nextDepth)
            sequence["relu" + str(index)] = nn.LeakyReLU(0.2, inplace=True)

            currDepth = nextDepth

        self.dimFeatureMap = currDepth

        self.main = nn.Sequential(sequence)
        self.main.apply(weights_init)

        self.initDecisionLayer(sizeDecisionLayer)

    def initDecisionLayer(self, sizeDecisionLayer):
        self.decisionLayer = nn.Conv2d(
            self.dimFeatureMap, sizeDecisionLayer, 4, 1, 0, bias=False)
        self.decisionLayer.apply(weights_init)
        self.sizeDecisionLayer = sizeDecisionLayer

    def forward(self, input, getFeature = False):
        x = self.main(input)

        if getFeature:

            return self.decisionLayer(x).view(-1, self.sizeDecisionLayer), \
                   x.view(-1, self.dimFeatureMap * 16)

        x = self.decisionLayer(x)
        return x.view(-1, self.sizeDecisionLayer)


def compute_gradient_penalty(D, real_images, fake_images):
    '''
    compute the gradient penalty where from the wgan-gp

    Args:
        D ([Moudle]): Discriminator
        real_images ([tensor]): real images from the dataset
        fake_images ([tensor]): fake images from teh G(z)

    Returns:
        [tensor]: computed the gradient penalty
    '''        
    # compute gradient penalty
    alpha = torch.rand(real_images.size(0), 1, 1, 1).cuda().expand_as(real_images)
    # (*, 1, 64, 64)
    interpolated = (alpha * real_images.data + ((1 - alpha) * fake_images.data)).requires_grad_(True)
    # (*,)
    out = D(interpolated)
    # get gradient w,r,t. interpolates
    grad = torch.autograd.grad(
        outputs=out,
        inputs = interpolated,
        grad_outputs = torch.ones(out.size()).cuda(),
        retain_graph = True,
        create_graph = True,
        only_inputs = True
    )[0]

    # grad_flat = grad.view(grad.size(0), -1)
    # grad_l2norm = torch.sqrt(torch.sum(grad_flat ** 2, dim=1))

    # grad_l2norm = torch.linalg.vector_norm(grad, ord=2, dim=[1,2,3]) # start version 1.10~
    grad_l2norm = grad.norm(2, dim=[1,2,3])

    gradient_penalty = torch.mean((grad_l2norm - 1) ** 2)

    return gradient_penalty
    
# ----------
#  Training
# ----------
def train():
    # Initialize generator and discriminator
    generator = Generator()
    discriminator = Discriminator()

    if os.path.exists("./models/wgan/generator.pth"):
        generator.load_state_dict(torch.load("./models/wgan/generator.pth"))
    if os.path.exists("./models/wgan/discriminator.pth"):
        discriminator.load_state_dict(torch.load("./models/wgan/discriminator.pth"))
        
    if cuda:
        generator.cuda()
        discriminator.cuda()
        
    # Configure data loader

    pos_data = Cup(
        path="./data/cup_plus/tensor_data/new",
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.CenterCrop(opt.img_size),  # 中心裁剪
                transforms.RandomHorizontalFlip(),  # 随机翻转
            ])
    )

    neg_data = Cup(
        path="./data/cup_plus/tensor_data/pos",
        transform=transforms.Compose(
            [
                transforms.Resize(opt.img_size),
                transforms.CenterCrop(opt.img_size),
                transforms.RandomHorizontalFlip(),
            ])
    )

    data = ConcatDataset([pos_data, neg_data])
    dataloader = torch.utils.data.DataLoader(
        pos_data,
        batch_size=opt.batch_size,
        shuffle=True,
    )

    # Optimizers
    optimizer_G = torch.optim.RMSprop(generator.parameters(), lr=opt.lr_g)
    optimizer_D = torch.optim.RMSprop(discriminator.parameters(), lr=opt.lr_d)

    # # Optimizers
    # optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr_g, betas=(opt.b1, opt.b2))
    # optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr_d, betas=(opt.b1, opt.b2))

   
    batches_done = 0
    for epoch in range(opt.n_epochs):

        for i, (imgs, _) in enumerate(dataloader):

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # 梯度归零
            optimizer_D.zero_grad()

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))
            # z = generate_noise(imgs.shape[0], opt.latent_dim)
            
            # Generate a batch of images
            fake_imgs = generator(z).detach()
            
            # Adversarial loss
            loss_D = -torch.mean(discriminator(real_imgs)) + torch.mean(discriminator(fake_imgs))
            # gradient_penalty
            grad = compute_gradient_penalty(discriminator, real_imgs, fake_imgs) * opt.gp_weight
            loss_D += grad
            
            loss_D.backward()
            optimizer_D.step()

            # Clip weights of discriminator
            # for p in discriminator.parameters():
            #    p.data.clamp_(-opt.clip_value, opt.clip_value)
            
            # Train the generator every n_critic iterations
            if i % opt.n_critic == 0:
                # -----------------
                #  Train Generator
                # -----------------

                optimizer_G.zero_grad()

                # Generate a batch of images
                gen_imgs = generator(z).detach()
                # Adversarial loss
                loss_G = -torch.mean(discriminator(gen_imgs))

                loss_G.backward()
                optimizer_G.step()

                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, batches_done % len(dataloader), len(dataloader), loss_D.item(), loss_G.item())
                )

            if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data, "images/wgan/%d.png" % batches_done, nrow=8, normalize=True)
                
            if batches_done % opt.model_interval == 0:
                torch.save(generator.state_dict(), f"models/wgan/generator.pth")
                torch.save(discriminator.state_dict(), f"models/wgan/discriminator.pth")
                
            batches_done += 1


def genRes():
    os.makedirs("images/results", exist_ok=True)
    generator = Generator()
    if cuda:
        generator.cuda()
    generator.load_state_dict(torch.load("./models/wgan/generator.pth"))
    n = 4800
    batch_size = 1
    row = 1
    for i in range(n):
        z = Variable(Tensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
        gen_imgs = generator(z).detach()
        save_image(gen_imgs.data, "images/results/%d.png" % i, nrow=row, normalize=True)
        print("images/results/%d.png" % i)
    print("Generate done.")
    
def main():
    train()

if __name__ == "__main__":
    main()