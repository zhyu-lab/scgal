import torch.nn as nn
import torch
from networks import Encoder, Decoder, Decoder_G,DiscriminatorA,GAN_AE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def Create_Model(opt): # define and create GAN_AE architecture
    A_col = opt.A_col
    B_col = opt.B_col
    encoder = Encoder(A_col, opt.latent_dim)
    encoder = encoder.to(device)
    decoder = Decoder(opt.latent_dim, A_col)
    decoder = decoder.to(device)
    decoder_G = Decoder_G(opt.latent_dim, B_col)
    decoder_G = decoder_G.to(device)
    netgan_ae = GAN_AE(encoder, decoder, decoder_G)
    netgan_ae = netgan_ae.to(device)
    netD_A = DiscriminatorA(B_col)
    netD_A = netD_A.to(device)
    return netgan_ae,netD_A

def set_requires_grad(nets, requires_grad=False):
        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

# Define loss function
class GANLoss(nn.Module):
    """
    The GANLoss class abstracts away the need to create the target label tensor that has the same size as the input.
    """
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """
        Initialize the GANLoss class.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """
        Create label tensors with the same size as the input.
        """
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """
        Calculate loss given Discriminator's output and grount truth labels.
        """
        if self.gan_mode in ['vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        return loss











