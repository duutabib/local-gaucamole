import torch
import utils 
import math
import random
import helpers
import numpy as np 
import matplotlib as mlp
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import transforms
from torch.utils.data import Subset
from torchvision.datasets import ImageFolder


# need to add the utils file to folder when submitting job

def compute_cvPCA(resp0):
    ss0 = utils.shuff_cvPCA(resp0, nshuff=10)
    ss0 = ss0.mean(axis=0)
    ss0 = ss0 / ss0.sum()

    return ss0

def get_activation(name):
    activation = {}
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def ghook(model, input, output):
    intermediate_activations = []
    return intermediate_activations.append(output)



# initialize model
model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
model.eval()


# set params for computation
num_rois = 193600
num_stimulus = 2800
arr00 = np.empty((2*num_stimulus, num_rois), dtype=float)
batch_size = 1
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
istim = np.empty(2*num_stimulus, dtype=int)
activation = {}

sample_target_layer = model.features[0]

# set the data
transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
         
            ]
        )

imagenet_data = ImageFolder('/camp/home/duuta/working/duuta/ppp0/data/ImageNet_data/imagenet-mini/', transform=transform)
imagenet_data2800 = Subset(imagenet_data, np.arange(2800))
data_loader = torch.utils.data.DataLoader(imagenet_data2800, 
                                          batch_size=batch_size,
                                          shuffle=False,
                                          num_workers=10) 

num_epochs = int(len(imagenet_data2800)/ batch_size)

for i in range(2):
    for epoch  in range(num_epochs):
        for j, (input, _) in enumerate(iter(data_loader)):
            hook_handle = sample_target_layer.register_forward_hook(get_activation('feats'))
            _ = model(input)
            arr00[i, j, :] = activation['feats'].reshape(1, num_rois)

hook_handle.remove()


# write activations to npy file
arr00.save('activation_for_conv1.npy', allow_pickle=True)