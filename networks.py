import torch
import torch.nn as nn
from torch.nn import init
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def init_weights(net, init_type):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                m.bias.data.zero_()

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>

class DiscriminatorA(nn.Module):
    """
    This class implements Discriminator
    """
    def __init__(self, inchannel):
        super(DiscriminatorA, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(inchannel, 32),
            nn.LeakyReLU(0.2, True),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32, 64),
            nn.LeakyReLU(0.2, True),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 1),
        )

        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
class Encoder(torch.nn.Module):
    """
    This class implements Encoder
    """
    def __init__(self, d_in, d_out):
        super(Encoder, self).__init__()
        self.enc_layer1 = nn.Sequential(
            nn.Linear(d_in, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.enc_layer2 = nn.Sequential(
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.enc_layer3 = nn.Sequential(
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.enc_layer4 = nn.Sequential(
            nn.Linear(128,64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.enc_layer5 = nn.Sequential(
            nn.Linear(64, d_out),
        )

        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.enc_layer1(x)
        x = self.enc_layer2(x)
        x = self.enc_layer3(x)
        x = self.enc_layer4(x)
        x = self.enc_layer5(x)
        return x

class Decoder(torch.nn.Module):
    """
    This class implements Decoder
    """
    def __init__(self, d_in, d_out):
        super(Decoder, self).__init__()
        self.dec_layer1 = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(),
        )
        self.dec_layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(),
        )
        self.dec_layer3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
        )
        self.dec_layer4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
        )
        self.dec_layer5 = nn.Sequential(
            nn.Linear(512, d_out),
            nn.LeakyReLU(),
        )

        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_layer3(x)
        x = self.dec_layer4(x)
        x = self.dec_layer5(x)
        return x


class Decoder_G(nn.Module):
    """
    This class implements Generator
    """
    def __init__(self, d_in, d_out):
        super(Decoder_G, self).__init__()
        self.dec_layer1 = nn.Sequential(
            nn.Linear(d_in, 64),
            nn.LeakyReLU(),
        )
        self.dec_layer2 = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(),
        )
        self.dec_layer3 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LeakyReLU(),
        )
        self.dec_layer4 = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(),
        )
        self.dec_layer5 = nn.Sequential(
            nn.Linear(512, d_out),
            nn.LeakyReLU(),
        )
        init_weights(self, init_type='kaiming')

    def forward(self, x):
        x = self.dec_layer1(x)
        x = self.dec_layer2(x)
        x = self.dec_layer3(x)
        x = self.dec_layer4(x)
        x = self.dec_layer5(x)
        return x
class GAN_AE(nn.Module):
    def __init__(self, encoder, decoder, decoder_G):
        super(GAN_AE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.decoder_G = decoder_G

    def forward(self, state):
        h_enc = self.encoder(state)
        rec = self.decoder(h_enc)
        fake = self.decoder_G(h_enc)
        return h_enc,rec,fake