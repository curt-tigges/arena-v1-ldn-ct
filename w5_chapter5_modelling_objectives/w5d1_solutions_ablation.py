# %%
import torch as t
from typing import Tuple
from torch import nn
from einops import rearrange
from einops.layers.torch import Rearrange
import os
import torchinfo
from tqdm import tqdm
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import wandb
import time
from dataclasses import dataclass
from torchvision import transforms, datasets
import sys

p = r"/home/curttigges/projects/arena-v1-ldn-ct"
# Replace the line above with your own root directory
os.chdir(p)
sys.path.append(p)
sys.path.append(p + r"/w5_chapter5_modelling_objectives")

device = t.device("cuda:0" if t.cuda.is_available() else "cpu")
# assert str(device) == "cuda:0"

import w5d1_utils
import w5d1_tests
from w0d2.solutions import pad1d, pad2d, conv1d_minimal, conv2d_minimal, Conv2d, Linear, ReLU, Pair, IntOrPair
from w0d3.solutions import BatchNorm2d, Sequential

MAIN = __name__ == "__main__"

def build_convtranspose_layers(
    n_layers,
    in_channels,
    out_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    batch_norm=True,
    activation=nn.ReLU(),
):
    """Builds a sequence of convolutional transpose layers with optional batch 
        normalization and activation layers.

    Args:
        n_layers (int): Number of convolutional transpose layers to build.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        output_padding (int): Output padding of the convolution.
        batch_norm (bool): Whether to use batch normalization layers.
        activation (nn.Module): Activation layer to use.

    Returns:
        nn.Sequential: A sequence of convolutional transpose layers.
    """
    layers = []
    in_channels = [in_channels // (2**i) for i in range(n_layers)]
    out_channels = in_channels[1:] + [3]
    for i in range(n_layers - 1):
        layers.append(
            nn.ConvTranspose2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels[i]))
        if activation is not None:
            layers.append(activation)

    # Last layer
    layers.append(
        nn.ConvTranspose2d(
            in_channels=in_channels[-1],
            out_channels=out_channels[-1],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    )
    layers.append(nn.Tanh())

    return nn.Sequential(*layers)

def build_conv_layers(
    n_layers,
    img_channels,
    generator_channels,
    kernel_size=4,
    stride=2,
    padding=1,
    batch_norm=True,
    activation=nn.LeakyReLU(0.2),
):
    """Builds a sequence of convolutional layers with optional batch 
        normalization and activation layers.

    Args:
        n_layers (int): Number of transpose layers to build.
        kernel_size (int): Size of the convolutional kernel.
        stride (int): Stride of the convolution.
        padding (int): Padding of the convolution.
        output_padding (int): Output padding of the convolution.
        batch_norm (bool): Whether to use batch normalization layers.
        activation (nn.Module): Activation layer to use.

    Returns:
        nn.Sequential: A sequence of convolutional transpose layers.
    """
    layers = []

    out_channels = [generator_channels // (2**i) for i in range(n_layers)][::-1]
    in_channels = ([img_channels] + out_channels[:-1])
    # First layer
    layers.append(
        nn.Conv2d(
            in_channels=in_channels[0],
            out_channels=out_channels[0],
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
    )
    layers.append(activation)
    
    for i in range(1, n_layers):
        layers.append(
            nn.Conv2d(
                in_channels=in_channels[i],
                out_channels=out_channels[i],
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False
            )
        )
        if batch_norm:
            layers.append(nn.BatchNorm2d(out_channels[i]))
        if activation is not None:
            layers.append(activation)

    return nn.Sequential(*layers)


class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
        scale_factor: int = 2,
    ):
        """Implementation of DCGAN generator

        Args:
            self (Generator): Module
            latent_dim_size (int): size of the random vector we use for generating output
            img_size (int): size of the images we're generating
            img_channels (int): indicates RGB images
            generator_num_features (int): number of channels after first projection and reshaping
            n_layers (int): number of CONV_n layers
            scale_factor (int): scale factor for upsampling
        """
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers
        print(f"{img_size=} {img_channels=} {generator_num_features=} {n_layers=}")
        self.initial = nn.Sequential(
            nn.Linear(
                latent_dim_size,
                generator_num_features * (img_size // scale_factor**self.n_layers) ** 2,
                bias=False,
            ),
            Rearrange(
                "b (c h w) -> b c h w", h=img_size // scale_factor**self.n_layers,
                w=img_size // scale_factor**self.n_layers
            ),
            nn.BatchNorm2d(generator_num_features),
            nn.ReLU(),
        )

        self.layers = build_convtranspose_layers(
            n_layers,
            in_channels=generator_num_features,
            out_channels=generator_num_features // 2,
            kernel_size=4,
            stride=2,
            padding=1,
            batch_norm=True,
            activation=nn.ReLU(),
        )

    def forward(self, x: t.Tensor):
        """Forward pass of the generator

        Args:
            self (Generator): Module
            x (t.Tensor): input tensor

        Returns:
            t.Tensor: generated image
        """
        x = self.initial(x)
        x = self.layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int,
        img_channels: int,
        generator_num_features: int,
        n_layers: int,
    ):
        """Implementation of DCGAN discriminator

        Args:
            img_size (int): _description_
            img_channels (int): _description_
            generator_num_features (int): _description_
            n_layers (int): _description_
        """        
        super().__init__()
        self.img_size = img_size
        self.img_channels = img_channels
        self.generator_num_features = generator_num_features
        self.n_layers = n_layers

        self.layers = build_conv_layers(n_layers, img_channels, generator_num_features)
        self.classifier = nn.Sequential(
            Rearrange("b c h w -> b (c h w)"),
            nn.Linear(generator_num_features * (img_size // 2**n_layers)**2, 1, bias=False),
            nn.Sigmoid(),
        )
        

    def forward(self, x: t.Tensor):
        """Forward pass of the discriminator

        Args:
            x (t.Tensor): input tensor

        Returns:
            t.Tensor: output tensor
        """
        x = self.layers(x)
        x = self.classifier(x)
        return x


def initialize_weights(model) -> None:
    """Initializes the weights of the model.

    Args:
        model (nn.Module): Model to initialize.
    """
    for m in model.modules():
        if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight, 1.0, 0.02)
            nn.init.constant_(m.bias, 0)

class DCGAN(nn.Module):
    """Implementation of DCGAN

    Args:
        nn (nn.Module): Parent class
    """    
    netD: Discriminator
    netG: Generator
    def __init__(self, args):
        """_summary_
        """        
        super().__init__()
        self.netG = Generator(
            latent_dim_size=args.latent_dim_size,
            img_size=args.img_size,
            img_channels=args.img_channels,
            generator_num_features=args.generator_num_features,
            n_layers=args.n_layers,
        )
        self.netD = Discriminator(
            img_size=args.img_size,
            img_channels=args.img_channels,
            generator_num_features=args.generator_num_features,
            n_layers=args.n_layers,
        )
        initialize_weights(self)

celeba_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 1024,
    n_layers = 4,
)
celeba_mini_config = dict(
    latent_dim_size = 100,
    img_size = 64,
    img_channels = 3,
    generator_num_features = 512,
    n_layers = 4,
)

#celeb_DCGAN = DCGAN(**celeba_config).to(device).train()
#celeb_mini_DCGAN = DCGAN(**celeba_mini_config).to(device).train()

# %%

# ======================== CELEB_A ========================

if MAIN:
    image_size = 64
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = ImageFolder(
        root="/home/curttigges/projects/arena-v1-ldn-ct/w5_chapter5_modelling_objectives/data",
        transform=transform
    )

    w5d1_utils.show_images(trainset, rows=3, cols=5)

# ======================== MNIST ========================

# img_size = 24

# from torchvision import datasets, transforms
# from torch.utils.data import DataLoader

# transform = transforms.Compose([
#     transforms.Resize(img_size),
#     transforms.ToTensor(),
#     transforms.Normalize((0.1307,), (0.3081,))
# ])

# trainset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)


# %%

@dataclass
class DCGANargs():
    latent_dim_size: int
    img_size: int
    img_channels: int
    generator_num_features: int
    n_layers: int
    trainset: datasets.ImageFolder
    batch_size: int = 8
    epochs: int = 1
    lr: float = 0.0002
    betas: Tuple[float] = (0.5, 0.999)
    track: bool = True
    cuda: bool = False
    seconds_between_image_logs: int = 40

def train_DCGAN(model, trainloader, args: DCGANargs):
    """Train DCGAN network

    Args:
        args (DCGANargs): _description_
    """
    
    if args.track:
        wandb.init(project="Curt-DCGAN", config=args)
        wandb.watch(model, log="all")

    optimizerD = t.optim.Adam(model.netD.parameters(), lr=args.lr, betas=args.betas)
    optimizerG = t.optim.Adam(model.netG.parameters(), lr=args.lr, betas=args.betas)

    last_log_time = time.time()

    for epoch in range(args.epochs):

        progress_bar = tqdm(trainloader)

        for img, _ in progress_bar:
            
            img = img.to(device)

            # Train discriminator
            optimizerD.zero_grad()
            noise = t.randn(args.batch_size, args.latent_dim_size).to(device)
            score_real = model.netD(img)
            img_gen = model.netG(noise)
            score_gen = model.netD(img_gen.detach())
            lossD = -(t.log(score_real).mean() + t.log(1 - score_gen).mean())
            lossD.backward()
            optimizerD.step()

            # Train generator
            optimizerG.zero_grad()
            score_gen = model.netD(img_gen)
            lossG = -t.log(score_gen).mean()
            lossG.backward()
            optimizerG.step()

            progress_bar.set_description(f"{epoch=}, lossD={lossD.item():.4f}, lossG={lossG.item():.4f}")

            if args.track:
                wandb.log(dict(lossD=lossD, lossG=lossG))
                if time.time() - last_log_time > args.seconds_between_image_logs:
                    last_log_time = time.time()
                    arrays = get_generator_output(model.netG) # shape (8, 64, 64, 3)
                    images = [wandb.Image(arr) for arr in arrays]
                    wandb.log({"images": images})
            
    return model

@t.inference_mode()
def get_generator_output(netG, n_examples=8, rand_seed=0):
    netG.eval()
    device = next(netG.parameters()).device
    t.manual_seed(rand_seed)
    noise = t.randn(n_examples, netG.latent_dim_size).to(device)
    arrays = rearrange(netG(noise), "b c h w -> b h w c").detach().cpu().numpy()
    netG.train()
    return arrays

# %%

if MAIN:
    args = DCGANargs(**celeba_mini_config, trainset=trainset)
    model = DCGAN(args).to(device).train()
    # print_param_count(model)
    x = t.randn(3, 100).to(device)
    statsG = torchinfo.summary(model.netG, input_data=x)
    statsD = torchinfo.summary(model.netD, input_data=model.netG(x))
    print(statsD)

# %%

if MAIN:
    args = DCGANargs(**celeba_mini_config, trainset=trainset)
    device = t.device("cuda" if args.cuda else "cpu")
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8)
    model = DCGAN(args).to(device)
    model = train_DCGAN(model, trainloader, args)
# %%