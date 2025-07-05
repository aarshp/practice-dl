import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import math
import pickle
import os

class Scheduler:
    def __init__(self, T, start, end, device):
        self.T = T
        self.start = start
        self.end = end
        self.beta = torch.linspace(start, end, T).to(device)
        self.alpha = 1 - self.beta
        self.alphaBar = torch.cumprod(self.alpha, dim=0).to(device)
        self.sqrtAlphaBar = torch.sqrt(self.alphaBar).to(device)
    
    def samplet(self, x0, t, noise):
        sqrtAlphaBart = self.sqrtAlphaBar[t].view(-1, *([1]*(x0.ndim-1)))
        sqrtOneMinusAlphaBart =  torch.sqrt(1.0 - sqrtAlphaBart**2)
        xt = sqrtAlphaBart*x0 + sqrtOneMinusAlphaBart*noise
        return xt
    
class ResnetBlock(nn.Module):
    def __init__ (self, inChannel, outChannel, positionalEncodingSize):
        super(ResnetBlock, self).__init__()

        self.doSkip = False
        self.doProjection = True
        if inChannel != outChannel:
            self.doSkip = True
            self.skipConv = nn.Conv2d(inChannel, outChannel, (1,1), stride=1)
        
        self.conv1 = nn.Conv2d(inChannel, outChannel, (3,3), stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outChannel, outChannel, (3,3), stride=1, padding=1, bias=False)
        self.projection = nn.Linear(positionalEncodingSize, outChannel)
        self.norm1 = nn.GroupNorm(8, outChannel)
        self.norm2 = nn.GroupNorm(8, outChannel)
        self.activation = nn.SiLU()
        
    def forward(self, input, positionalEncoding):
        output = self.conv1(input)
        output = self.norm1(output)
        output = self.activation(output)
        projectedEncoding = self.projection(positionalEncoding)
        projectedEncoding = projectedEncoding.unsqueeze(2).unsqueeze(3).to(input.device)
        output = output + projectedEncoding
        output = self.conv2(output)
        output = self.norm2(output)
        output = self.activation(output)

        if (self.doSkip == True):
            output = output + self.skipConv(input)
        else:
            output = output + input

        return output
    

class DenoiserUnet(nn.Module):
    def __init__(self, postionalEncodingSize = 128, inputChannel = 1, startingChannel = 8, stages =3) :
        super().__init__()
        self.postionalEncodingSize = postionalEncodingSize
        self.startingChannel = startingChannel
        self.stemConv = nn.Conv2d(inputChannel, startingChannel, (3,3), 1, 1)
        self.downsamplingBlocks = nn.ModuleList()
        for stage in range(stages):
            channelSize = (int)(startingChannel*(2**(stage)))
            self.downsamplingBlocks.append(ResnetBlock(channelSize, channelSize, postionalEncodingSize))
            self.downsamplingBlocks.append(ResnetBlock(channelSize, channelSize, postionalEncodingSize))
            self.downsamplingBlocks.append(nn.Conv2d(channelSize, channelSize*2, (3,3), stride=2, padding=1))

        channelSize = (int)(startingChannel*(2**(stages)))
  
        self.bottleneck = nn.ModuleList([ResnetBlock(channelSize, channelSize, postionalEncodingSize), 
                                         ResnetBlock(channelSize, channelSize, postionalEncodingSize)])
        self.upsamplingBlocks = nn.ModuleList()
        for stage in range(stages):
            channelSize = (int)(startingChannel*(2**(stages-stage)))
            self.upsamplingBlocks.append(nn.ConvTranspose2d(channelSize, channelSize//2, (3,3), stride=2, padding=1, output_padding=1))
            self.upsamplingBlocks.append(ResnetBlock(channelSize, channelSize//2, postionalEncodingSize))
            self.upsamplingBlocks.append(ResnetBlock(channelSize//2, channelSize//2, postionalEncodingSize))
        
        self.stemUpsampleConv = nn.Conv2d(startingChannel, inputChannel, (3,3), 1, 1)

    def forward(self, xt, t):
        device = xt.device
        tTensor = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(-1)  
        divTerm = torch.exp(
            torch.arange(self.postionalEncodingSize // 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / (self.postionalEncodingSize // 2))
        )

        angles = tTensor * divTerm 

        positionalEncoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)  

        output = self.stemConv(xt)
        skipOutputs = []
        for idx,layer in enumerate(self.downsamplingBlocks):
            if idx%3 == 2:
                output = layer(output) #downsample conv
            else:
                output = layer(output, positionalEncoding) #resnet blocks
            if idx%3 == 1:
                skipOutputs.append(output) #output of second resnet
        
        for layer in self.bottleneck:
            output = layer(output, positionalEncoding)
        
        for idx,layer in enumerate(self.upsamplingBlocks):
            if idx %3 ==0:
                output = layer(output)
                output = torch.cat((output, skipOutputs.pop()), dim=1)
            else:
                output = layer(output, positionalEncoding)

        output = self.stemUpsampleConv(output)
        return output
        
    
def visualizeMNIST(dataset):
    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    for i in range(25):
        image, label = dataset[i]
        ax = axes[i // 5, i % 5]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title(f"{label}", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def visualizeImages(images):
    images = images.to('cpu')
    gridSize = (int)(math.sqrt((images.shape[0])))

    fig, axes = plt.subplots(gridSize, gridSize, figsize=(8, 8))

    for i in range(gridSize*gridSize):
        image = images[i]
        ax = axes[i // gridSize, i % gridSize]
        ax.imshow(image.squeeze(), cmap='gray')
        ax.set_title("hehe", fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def loadMNISTDataset(device, train=True):
    if train:
        pickle_path = 'mnist_dataset_32x32.pkl'
    else:
        pickle_path  = 'mnist_dataset_test_32x32.pkl'

    if os.path.exists(pickle_path):
        print(f"Loading dataset from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            mnistDataset = pickle.load(f)
    else:
        print("Downloading and preparing dataset...")
        transform = transforms.Compose([
            transforms.Resize((32, 32)),  
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  
        ])
        mnistDataset = datasets.MNIST('.', download=True, transform=transform, train=train)
        
        print(f"Saving dataset to {pickle_path}...")
        with open(pickle_path, 'wb') as f:
            pickle.dump(mnistDataset, f)

    return mnistDataset

def train(device, dataset, denoiser, scheduler, optimizer=None, epochs = 20):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
    lossFn = nn.MSELoss()
    if optimizer ==  None: 
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-4)
    
    denoiser.train()

    for epoch in range(epochs):
        epochLoss = 0.0
        for i, (x, _) in enumerate(dataloader):
            batchSize = x.shape[0]
            x = x.to(device)
            t = torch.randint(0, scheduler.T, (batchSize,), dtype=int, device=device)
            noise = torch.randn_like(x)
            xt = scheduler.samplet(x, t, noise)
            predictedNoise = denoiser(xt, t)
            loss = lossFn(predictedNoise, noise)
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}") 
        
        epochLoss = epochLoss / len(dataloader)
        print(f"--- Epoch {epoch} Average Loss: {epochLoss:.4f} ---")

        if epoch % 4 == 0:
            torch.save(denoiser.state_dict(), f'denoiser_model_{epoch}.pth')
            print(f"Model saved at epoch {epoch} with new best loss: {epochLoss:.4f}")


def generateImages(denoiser, scheduler, modelPath, device, nSteps = 5, nSamples=5, imageSize =(1,32,32)):
    
    denoiser.load_state_dict(torch.load(modelPath, device))
    denoiser.eval()
    collectedXs= []
    with torch.no_grad():
        x  = torch.randn(nSamples, *imageSize, device=device)
        for step in range(nSteps - 1, -1, -1):
            
            t = torch.full((nSamples,), step, dtype=int, device=device)
            predictedNoise = denoiser(x, t)
            alphat = scheduler.alpha[t].view(-1, *([1]*(len(imageSize))))
            alphaBart = scheduler.alphaBar[t].view(-1, *([1]*(len(imageSize))))
            betat = scheduler.beta[t].view(-1, *([1]*(len(imageSize))))
            alphaBartMinusOne = scheduler.alphaBar[t-1].view(-1, *([1]*(len(imageSize))))
            muAtT = (1/torch.sqrt(alphat))*(x - (betat/torch.sqrt(1-alphaBart))*predictedNoise)
            sigmaAtT = betat*(1-alphaBartMinusOne)/(1-alphaBart)
            if step == 0:
                x = muAtT
            else:
                x = muAtT + torch.sqrt(sigmaAtT)*torch.randn_like(x)
            collectedXs.append(x)

    
    visualizeImages(x)
    collectedXs = torch.stack(collectedXs)
    firstImageSteps = collectedXs[::40, 0, :, :, :]
    visualizeImages(firstImageSteps)
    return x
    

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    mnistDataset = loadMNISTDataset(device)
    scheduler = Scheduler(start=1e-4, end=0.02, T=1000, device=device) 
    denoiser = DenoiserUnet().to(device)
    # train(device, mnistDataset, denoiser, scheduler)

    # images = generateImages(denoiser, scheduler, "./denoiser_model_16.pth", device, nSteps=1000, nSamples=25)

if __name__ == '__main__':
    main()