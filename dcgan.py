import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
def weights_init(w):
    """
    Initializes the weights of the layer, w.
    """
    classname = w.__class__.__name__
    if classname.find('conv') != -1:
        nn.init.normal_(w.weight.data, 0.0, 0.02)
    elif classname.find('bn') != -1:
        nn.init.normal_(w.weight.data, 1.0, 0.02)
        nn.init.constant_(w.bias.data, 0)

# Define the Generator Network
class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 280 x 140
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
             4, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(params['ndf'])

        # Input Dimension: (ndf) x 140 x 70
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
             4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
             4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        self.tconv1 = nn.ConvTranspose2d(params['ndf']*4, params['ndf']*2,
             4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ngf*2) x 70 x 35
        self.tconv2 = nn.ConvTranspose2d(params['ngf']*2, params['ngf'],
            4, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(params['ngf'])

        # Input Dimension: (ngf) * 140 * 70
        self.tconv3 = nn.ConvTranspose2d(params['ngf'], params['nc'],
            4, 2, 1, bias=False)
        #Output Dimension: (nc) x 280 x 140

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.tconv1(x)))
        x = F.relu(self.bn5(self.tconv2(x)))
        x = torch.tanh(self.tconv3(x))
        return x

# Define the Discriminator Network
class Discriminator(nn.Module):
    def __init__(self, params):
        super().__init__()

        # Input Dimension: (nc) x 64 x 64
        self.conv1 = nn.Conv2d(params['nc'], params['ndf'],
            4, 2, 1, bias=False)

        # Input Dimension: (ndf) x 32 x 32
        self.conv2 = nn.Conv2d(params['ndf'], params['ndf']*2,
            4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(params['ndf']*2)

        # Input Dimension: (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(params['ndf']*2, params['ndf']*4,
            4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(params['ndf']*4)

        # Input Dimension: (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(params['ndf']*4, params['ndf']*8,
            4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(params['ndf']*8)

        # Input Dimension: (ndf*8) x 4 x 4
        self.conv5 = nn.Conv2d(params['ndf']*8, 1, 4, 1, 0, bias=False)



    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), 0.2, True)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2, True)
        x = F.relu(self.conv5(x))
        x = F.relu(nn.Linear(70, 40).to(device)(torch.flatten(x, start_dim=1)))
        x = F.relu(nn.Linear(40, 10).to(device)(x))
        x = torch.sigmoid((nn.Linear(10, 1).to(device)(x)))
        return x
    
if __name__ == "__main__":
    
    device = torch.device("cuda:0" if(torch.cuda.is_available()) else "cpu")
    print(device, " will be used.\n")
    
    params = {
    "file_path" : 'train.npy',
    "mask" : 'mask.npy',
    "bsize" : 256,# Batch size during training.
    "height" : 140,# height of image
    "width" : 280,# width of image
    'nc' : 1,# Number of channles in the training images. For coloured images this is 3.
    'nz' : 100,# Size of the Z latent vector (the input to the generator).
    'ngf' : 64,# Size of feature maps in the generator. The depth will be multiples of this.
    'ndf' : 64, # Size of features maps in the discriminator. The depth will be multiples of this.
    'nepochs' : 10,# Number of training epochs.
    'lr' : 0.0002,# Learning rate for optimizers
    'beta1' : 0.5,# Beta1 hyperparam for Adam optimizer
    'save_epoch' : 1}# Save step.
    
    netG = Generator(params)
    netG.apply(weights_init)
    netG = netG.to(device)
    print(netG)
    
    fake_data = netG(masked_data).to(device)
    fake_data = (1.-mask)*fake_data + masked_data
    
    train = np.load(params['file_path'])/255
    train = torch.from_numpy(train)
    mask = np.load(params['mask'])
    mask =  torch.from_numpy(mask)
    
    masked_img = train[0]*mask
    print(masked_img.shape)
    with torch.no_grad():
        generated_img = netG(masked_img.reshape(1, 1, 140, 280).float()).detach()
        generated_img = (1.-mask)*generated_img + masked_img
    print(generated_img.shape)

    
