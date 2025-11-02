import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import math
import pickle
import os
import random

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
    def __init__ (self, inChannel, outChannel, encodingSize):
        super(ResnetBlock, self).__init__()

        self.doProjectedSkip = False
        if inChannel != outChannel:
            self.doProjectedSkip = True
            self.projectedSkip = nn.Conv2d(inChannel, outChannel, (1,1), stride=1)
        
        self.conv1 = nn.Conv2d(inChannel, outChannel, (3,3), stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(outChannel, outChannel, (3,3), stride=1, padding=1, bias=False)
        self.projection = nn.Linear(encodingSize, outChannel)
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

        if (self.doProjectedSkip == True):
            output = output + self.projectedSkip(input)
        else:
            output = output + input

        return output
    
class LeNet5(nn.Module):
    def __init__(self, inChannel, nClasses):
        super(LeNet5, self).__init__()
    
        self.convSequential = nn.Sequential(
            nn.Conv2d(inChannel, 6, (5,5), stride=1),
            nn.GroupNorm(1,6),
            nn.ReLU(),
            nn.MaxPool2d((2,2), stride=2),
            nn.Conv2d(6, 16, (5,5), stride=1),
            nn.GroupNorm(1, 16),
            nn.ReLU(),
            nn.AvgPool2d((2,2), stride=2),
            nn.SiLU()
        )
        self.fcSeq = nn.Sequential(
            nn.Linear(400, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, nClasses),
        )

    def forward(self, xt, scaledTimeStep ):
        timeStepTensor = scaledTimeStep.view(-1, 1, 1, 1)
        timeStepTensor = timeStepTensor.expand(-1, 1, xt.shape[2], xt.shape[3])
        out = torch.cat((xt, timeStepTensor), dim=1)
        out = self.convSequential(out)
        out = out.reshape(out.size(0), -1)
        out = self.fcSeq(out)
        return out

class DenoiserUnet(nn.Module):
    def __init__(self, timeEncodingSize = 128, inputChannel = 1, startingChannel = 8, stages = 3, noOfClasses = None) :
        super().__init__()
        self.postionalEncodingSize = timeEncodingSize
        self.startingChannel = startingChannel
        self.stemConv = nn.Conv2d(inputChannel, startingChannel, (3,3), 1, 1)
        self.addClassCondition = False
        if noOfClasses != None:
            self.addClassCondition = True
            self.classEmbeddingLayer = nn.Embedding(noOfClasses + 1, timeEncodingSize)
        self.downsamplingBlocks = nn.ModuleList()
        for stage in range(stages):
            channelSize = (int)(startingChannel*(2**(stage)))
            self.downsamplingBlocks.append(ResnetBlock(channelSize, channelSize, timeEncodingSize))
            self.downsamplingBlocks.append(ResnetBlock(channelSize, channelSize, timeEncodingSize))
            self.downsamplingBlocks.append(nn.Conv2d(channelSize, channelSize*2, (3,3), stride=2, padding=1))

        channelSize = (int)(startingChannel*(2**(stages)))
  
        self.bottleneck = nn.ModuleList([ResnetBlock(channelSize, channelSize, timeEncodingSize), 
                                         ResnetBlock(channelSize, channelSize, timeEncodingSize)])
        self.upsamplingBlocks = nn.ModuleList()
        for stage in range(stages):
            channelSize = (int)(startingChannel*(2**(stages-stage)))
            self.upsamplingBlocks.append(nn.ConvTranspose2d(channelSize, channelSize//2, (3,3), stride=2, padding=1, output_padding=1))
            self.upsamplingBlocks.append(ResnetBlock(channelSize, channelSize//2, timeEncodingSize))
            self.upsamplingBlocks.append(ResnetBlock(channelSize//2, channelSize//2, timeEncodingSize))
        
        self.stemUpsampleConv = nn.Conv2d(startingChannel, inputChannel, (3,3), 1, 1)

    def forward(self, xt, t, y = None):
        device = xt.device
        tTensor = torch.as_tensor(t, dtype=torch.float32, device=device).unsqueeze(-1)  
        divTerm = torch.exp(
            torch.arange(self.postionalEncodingSize // 2, device=device, dtype=torch.float32)
            * (-math.log(10_000.0) / (self.postionalEncodingSize // 2))
        )

        angles = tTensor * divTerm 
        timeEncoding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
        if self.addClassCondition:
            classEncoding = self.classEmbeddingLayer(y)
            finalEncoding = timeEncoding + classEncoding
        else:
            finalEncoding = timeEncoding  
        output = self.stemConv(xt)
        skipOutputs = []
        for idx,layer in enumerate(self.downsamplingBlocks):
            if idx%3 == 2:
                output = layer(output) #downsample conv
            else:
                output = layer(output, finalEncoding) #resnet blocks
            if idx%3 == 1:
                skipOutputs.append(output) #output of second resnet
        
        for layer in self.bottleneck:
            output = layer(output, finalEncoding)
        
        for idx,layer in enumerate(self.upsamplingBlocks):
            if idx %3 ==0:
                output = layer(output)
                output = torch.cat((output, skipOutputs.pop()), dim=1)
            else:
                output = layer(output, finalEncoding)

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

def visualizeImages(images, sampleClasses=None):
    images = images.to('cpu')
    gridSize = (int)(math.sqrt((images.shape[0])))

    fig, axes = plt.subplots(gridSize, gridSize, figsize=(8, 8))

    for i in range(gridSize*gridSize):
        image = images[i]
        ax = axes[i // gridSize, i % gridSize]
        ax.imshow(image.squeeze(), cmap='gray')
        if sampleClasses is None:
            ax.set_title("hehe", fontsize=10)
        else:
            ax.set_title(sampleClasses[i].item(),  fontsize=10)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def loadCifarDataset(device, train = True):
    if train:
        pickle_path = 'cifar_dataset_32x32.pkl'
    else:
        pickle_path = 'cifar_dataset_test_32x32.pkl'

    if os.path.exists(pickle_path):
        print(f"Loading dataset from {pickle_path}...")
        with open(pickle_path, 'rb') as f:
            cifarDataset = pickle.load(f)
    else:
        print("Downloading and preparing Cifar Dataset...")
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])
        cifarDataset  = datasets.CIFAR10('.', download=True, transform=transform, train=train)
    return cifarDataset

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

def trainClassifier(dataset, scheduler, classifier, device, optimizer = None, epochs = 5):

    dataloader = DataLoader (dataset, shuffle=True, batch_size=128)
    lossFn = nn.CrossEntropyLoss()
    if optimizer ==  None: 
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
    noiseLevelMap = {
        0:int(scheduler.T*0.3),
        1:int(scheduler.T*0.3),
        2:int(scheduler.T*0.6),
        3:int(scheduler.T*0.6),
        4:int(scheduler.T*0.8),
        5:int(scheduler.T)
    }
    for epoch in range(max(epochs, len(noiseLevelMap))):
        epochLoss = 0.0
        for i, (x, label) in enumerate(dataloader):
            batchSize = x.shape[0]
            x = x.to(device)
            label = label.to(device)
            noiseLevel = scheduler.T if epoch not in noiseLevelMap else noiseLevelMap[epoch]
            t = torch.randint(0, noiseLevel, (batchSize,), dtype=torch.long, device=device)
            noise = torch.randn_like(x)
            xt = scheduler.samplet(x, t, noise) 
            predictedLogits = classifier(xt, t.float()/scheduler.T)
            loss = lossFn(predictedLogits, label)
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}") 
        
        epochLoss = epochLoss / len(dataloader)
        print(f"--- Epoch {epoch} Average Loss: {epochLoss:.4f} ---")

        torch.save(classifier.state_dict(), f'classifier_model_{epoch}.pth')
        print(f"Model saved at epoch {epoch} with new best loss: {epochLoss:.4f}")


def trainDenoiser(device, dataset, denoiser, scheduler, optimizer=None, epochs = 20, percentageDropLabel = None, checkpointPath = None):
    dataloader = DataLoader(dataset, shuffle=True, batch_size=64)
    lossFn = nn.MSELoss()
    if optimizer ==  None: 
        optimizer = torch.optim.Adam(denoiser.parameters(), lr=1e-4)
    startingEpoch = 0

    if checkpointPath != None:
        denoiser.load_state_dict(torch.load(checkpointPath, device))
        startingEpoch = checkpointPath.split('_')[-1].split('.')[0]
        startingEpoch = int(startingEpoch)
        print(f"Loaded checkpoint from epoch: {startingEpoch}")
    denoiser.train()

    for epoch in range(startingEpoch, startingEpoch + epochs):
        epochLoss = 0.0
        for i, (x, y) in enumerate(dataloader):
            batchSize = x.shape[0]
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, scheduler.T, (batchSize,), dtype=int, device=device)
            noise = torch.randn_like(x)
            xt = scheduler.samplet(x, t, noise)
            if percentageDropLabel is not None:
                mask = torch.rand(y.shape, device=y.device) < percentageDropLabel/100.0
                y [mask] = 10
            else:
                y = None
            predictedNoise = denoiser(xt, t, y)                
            loss = lossFn(predictedNoise, noise)
            epochLoss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if i%100 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}") 
        
        epochLoss = epochLoss / len(dataloader)
        print(f"--- Epoch {epoch} Average Loss: {epochLoss:.4f} ---")

        if epoch % 10 == 0:
            torch.save(denoiser.state_dict(), f'denoiser_model_{epoch}.pth')
            print(f"Model saved at epoch {epoch} with new best loss: {epochLoss:.4f}")


def generateDDPMSampleImages(denoiser, scheduler, device, nSteps = 10000, nSamples=5, imageSize =(1,32,32), classifier = None, guidanceFactor = 7):
    denoiserModel, denoiserParamsPath = denoiser
    denoiserModel.load_state_dict(torch.load(denoiserParamsPath, device))
    denoiserModel.eval()
    generateImagesWithClassifierGuidance = classifier != None
    if generateImagesWithClassifierGuidance:
        classifierModel, classifierParamPath = classifier
        classifierModel.load_state_dict(torch.load(classifierParamPath, device))
        sampleClasses = torch.randint(0, 10, (nSamples, ), dtype=int, device=device)
        classifierModel.eval()
        
    collectedXs= []
    x  = torch.randn(nSamples, *imageSize, device=device)
    
    for step in range(nSteps - 1, -1, -1):
        
        t = torch.full((nSamples,), step, dtype=int, device=device)
        with torch.no_grad():
            predictedNoise = denoiserModel(x, t)
        if generateImagesWithClassifierGuidance:
            xIn = x.detach().requires_grad_(True)
            logp = F.log_softmax(classifierModel(xIn, t.float()/scheduler.T), dim=-1)
            classifierLoss = logp[torch.arange(xIn.shape[0]), sampleClasses].sum()
            grad = torch.autograd.grad(classifierLoss, xIn)[0]
            x.requires_grad = False
        else:
            grad = torch.zeros_like(x)

        alphat = scheduler.alpha[t].view(-1, *([1]*(len(imageSize))))
        alphaBart = scheduler.alphaBar[t].view(-1, *([1]*(len(imageSize))))
        betat = scheduler.beta[t].view(-1, *([1]*(len(imageSize))))
        if step > 0:
            alphaBartMinusOne = scheduler.alphaBar[t-1].view(-1, *([1]*(len(imageSize))))
        else:
            alphaBartMinusOne = torch.ones_like(alphaBart)
        sigmaAtT = betat*(1-alphaBartMinusOne)/(1-alphaBart)

        guidedPredictedNoise = predictedNoise - guidanceFactor*grad*sigmaAtT            
        muAtT = (1/torch.sqrt(alphat))*(x - (betat/torch.sqrt(1-alphaBart))*guidedPredictedNoise)
        if step == 0:
            x = muAtT
        else:
            x = muAtT + torch.sqrt(sigmaAtT)*torch.randn_like(x)
        collectedXs.append(x)

    
    visualizeImages(x, sampleClasses)
    collectedXs = torch.stack(collectedXs)
    firstImageSteps = collectedXs[::40, 0, :, :, :]
    visualizeImages(firstImageSteps)
    return x
    

def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    # mnistDataset = loadMNISTDataset(device)
    scheduler = Scheduler(start=1e-4, end=0.02, T=1000, device=device) 
    # denoiser = DenoiserUnet().to(device)
    # trainDenoiser(device, mnistDataset, denoiser, scheduler)    
    classifier = LeNet5(inChannel=2, nClasses=10).to(device)
    # trainClassifier(mnistDataset, scheduler, classifier, device)
    cifarDataset = loadCifarDataset(device=device)

    cifarDenoiser  = DenoiserUnet(inputChannel=3, startingChannel=64).to(device)
    trainDenoiser(device, cifarDataset, cifarDenoiser, scheduler, epochs=51, percentageDropLabel=20, checkpointPath="./denoiser_model_40.pth")


    # images = generateDDPMSampleImages((denoiser,"./denoiser_model_16.pth"), scheduler,  device, nSteps=1000, nSamples=25, classifier=(classifier, "./classifier_model_5.pth"))
    

if __name__ == '__main__':
    main()