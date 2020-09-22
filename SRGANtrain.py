## IMPORTS
import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
import torchvision.models.vgg as vgg_models
from torchvision.utils import save_image, make_grid
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from PIL import Image
import numpy as np
import pkbar
import glob
import shutil
import os
from tqdm import tqdm


## HYPERPARAMETERS
lr = 0.0002
b1 = 0.9
b2 = 0.999
batch_size = 5
epoch = 1
n_epochs = 20
img_hr_shape = (3,512,512)
img_lr_shape = (3,512//4,512//4)
n_filters = 64
n_res_blocks = 16
n_discrim_blocks = 5

## CHECK CUDA AVAILABLE
cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

class celebADataset(Dataset):
    """collect celebA dataset."""

    def __init__(self, root_dir):

        self.transform = transforms.Compose([
            transforms.Resize([512, 512]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])

        self.target_transform = transforms.Compose([
            transforms.Resize([512//4, 512//4]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],std=[0.5, 0.5, 0.5])
            ])

        self.root_dir = root_dir

        ## dataset presorted into male and female folders, just need all photos unlabled
        self.files = sorted(glob.glob(self.root_dir + "**/*.jpg",recursive=True))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        raw_img = Image.open(self.files[idx % len(self.files)])
        img_hr = self.transform(raw_img)
        img_lr = self.target_transform(raw_img)

        return img_hr, img_lr

## INPUT DATASET
dataset = celebADataset('../input/celebahq/celeba_hq/train/')
data_loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=0)

## CHECK DATA SHAPES
for batch in data_loader:
    img_hr, img_lr = batch
    print(img_hr.shape)
    print(img_lr.shape)
    print(torch.max(img_hr))
    print(torch.min(img_hr))
    break

## CREATE A RESIDUAL BLOCK FOR THE GENERATOR
class ResidualBlock(nn.Module):
    def __init__(self, n_filters):
        super(ResidualBlock, self).__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters, 0.8),
            nn.PReLU(),
            nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(n_filters, 0.8),
        )

    def forward(self, x):
        return x + self.conv_block(x)


## MAKE GENERATOR
class make_generator(nn.Module):

    def __init__(self):
        super(make_generator, self).__init__()

        self.n_filters = n_filters

        res_blocks = []
        for _ in range(n_res_blocks):
            res_blocks.append(ResidualBlock(self.n_filters))
        self.res_blocks = nn.Sequential(*res_blocks)

        self.conv1 = nn.Conv2d(3, self.n_filters, 1, 1)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters, 1, 1)
        self.conv3 = nn.Conv2d(self.n_filters, 256, 1, 1)
        self.conv3b = nn.Conv2d(256, 3, 1, 1)
        self.conv4 = nn.Conv2d(256, 3, 1, 1)

        self.deconv1 = nn.ConvTranspose2d(self.n_filters, self.n_filters, 2, stride=2)
        self.deconv2 = nn.ConvTranspose2d(256, 256, 2, stride=2)


    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.res_blocks(x)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.conv3(x))

        x = F.relu(self.deconv2(x))
        x = torch.tanh(self.conv4(x))
        return x

## CREATE A DISCRIMINATOR BLOCK
class discrimBlock(nn.Module):
    def __init__(self, n_filt):
        super(discrimBlock, self).__init__()
        self.discrim_block = nn.Sequential(
            nn.Conv2d(n_filt, n_filt*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*2, 0.8),
            nn.PReLU(),
            nn.Conv2d(n_filt*2, n_filt*2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(n_filt*2, 0.8),
            nn.PReLU()
        )

    def forward(self, x):
        return self.discrim_block(x)


## CREATE THE DISCRIMINATOR
class make_discriminator(nn.Module):

    def __init__(self):
        super(make_discriminator, self).__init__()

        self.n_filters = n_filters

        self.conv1 = nn.Conv2d(3, self.n_filters, 1, 1)
        self.conv2 = nn.Conv2d(self.n_filters, self.n_filters*2, 1, 2)

        discrim_blocks = []
        for _ in range(n_discrim_blocks):
            discrim_blocks.append(discrimBlock(self.n_filters))
            self.n_filters *= 2
        self.discrim_blocks = nn.Sequential(*discrim_blocks)



        self.flat = nn.Flatten(1,-1)

        self.fc1 = nn.Linear(2048,1024)
        self.fc2 = nn.Linear(1024,1)



    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.discrim_blocks(x)
        x = self.flat(x)
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x

## MAKE THE FEATURE EXTRACTOR
class make_featureExtractor(nn.Module):

    def __init__(self):
        super(make_featureExtractor, self).__init__()

        self.vgg19 = vgg_models.vgg19(pretrained=True).features[:-2]

    def forward(self, x):
        x = self.vgg19(x)
        return x

## INIT MODELS
generator = make_generator().cuda()
discriminator = make_discriminator().cuda()
## FEATURE EXTRACTOR IN EVAL MODE SO UNTRAINABLE
feature_extractor = make_featureExtractor().cuda().eval()

## PRELOAD MODELS IF NOT THE FIRST EPOCH
if epoch != 0:
    generator.load_state_dict(torch.load("../input/celebhqmodels/celebaHQ_generator.pth"))
    discriminator.load_state_dict(torch.load("../input/celebhqmodels/celebaHQ_discriminator.pth"))

## LOSS FUNCTIONS AND OPT
mse_loss = torch.nn.MSELoss()
l1_loss = torch.nn.L1Loss()
optimizer_G = torch.optim.Adam(generator.parameters(), lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr, betas=(b1, b2))

for epoch in range(epoch, n_epochs):

    ## SET UP PROGRESS BAR
    kbar = pkbar.Kbar(target=len(data_loader), epoch=epoch, num_epochs=n_epochs, width=8, always_stateful=False)
    stateful_metrics=["G LOSS", "D LOSS"]

    for i, imgs in enumerate(data_loader):

        ## LOAD IMAGES
        img_hr, img_lr = imgs

        ## CONFIGURE THE MODEL OUTPUTS
        imgs_lr = Variable(img_lr.type(Tensor))
        imgs_hr = Variable(img_hr.type(Tensor))

        ## CONFIGURE GROUND TRUTHS
        valid = Variable(Tensor(np.ones((imgs_lr.size(0), 1))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((imgs_lr.size(0), 1))), requires_grad=False)

        ## TRAIN THE GENERATOR
        # Zero gradients
        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        fake_hr = generator(imgs_lr)


        # Adversarial loss
        ad_loss = mse_loss(discriminator(fake_hr), valid)

        # Content loss
        fake_features = feature_extractor(fake_hr)
        real_features = feature_extractor(imgs_hr)
        content_loss = l1_loss(fake_features, real_features.detach())

        # Total loss
        loss_G = content_loss + 1e-3 * ad_loss

        # Update generator
        loss_G.backward()
        optimizer_G.step()


        ## TRAIN DISCRIMINATOR
        # Zero gradients
        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = mse_loss(discriminator(imgs_hr), valid)
        loss_fake = mse_loss(discriminator(fake_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) * 0.5

        # Update discriminator
        loss_D.backward()
        optimizer_D.step()

        # UPDATE PROG BAR
        kbar.update(i, values=[("G LOSS", loss_G), ("D LOSS", loss_D)])

    ## AT EPOCH END SAVE A SAMPLE IMAGE TRANSFORMATION
    img_lr = nn.functional.interpolate(img_lr, scale_factor=4)
    fake_hr = make_grid(fake_hr[0], nrow=1, normalize=True).cpu()
    img_lr = make_grid(img_lr[0], nrow=1, normalize=True).cpu()
    img_hr = make_grid(img_hr[0], nrow=1, normalize=True).cpu()
    ## LOW RES // FAKE // HIGH RES
    img_grid = torch.cat((img_lr, fake_hr, img_hr), -1)
    save_image(img_grid, "celebaHQ_EPOCH_END_%d.png"%(epoch), normalize=False)


    ## SAVE MODELS
    torch.save(generator.state_dict(), "celebaHQ_generator.pth")
    torch.save(discriminator.state_dict(), "celebaHQ_discriminator.pth")
