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

    def process_one_image(self, image, WW, HH):
        image = image.cuda()

        """get one prediction"""
        pred = self.generator(image, training=False)
        pred = pred.sigmoid()
        pred = F.upsample(pred, size=[WW, HH], mode='bilinear', align_corners=False)
        pred = pred.sigmoid().data.cpu().numpy().squeeze()
        pred = 255 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)

        return pred
