import torch
import torch.nn.functional as F
from torch.autograd import Variable
from model.ResNet_models import Generator, FCDiscriminator
from data import test_dataset

class SamplingBVM():
    def __init__(self, model_path, channel=32, latent_dim=8):
        generator = Generator(channel=channel, latent_dim=latent_dim)
        generator.load_state_dict(torch.load(model_path))

        # nn. Module. cuda() moves all model parameters and buffers to the GPU.
        generator.cuda()
        # eval() is a PyTorch method that sets a model to evaluation mode 
        # explicitly. In evaluation mode, the model behaves differently 
        # from training mode, and several layers, such as dropout layers, 
        # batch normalization layers, and others, behave differently.
        generator.eval()
        self.generator = generator

        dis_model = FCDiscriminator(ndf=64)
        dis_model.load_state_dict(torch.load(model_path))
        dis_model.cuda()
        dis_model.eval()
        self.dis_model = dis_model