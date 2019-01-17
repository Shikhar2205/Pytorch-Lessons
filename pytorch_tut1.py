from torchvision import datasets,transforms
import torch
import matplotlib.pyplot as plt
transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5)),])

trainset=datasets.MNIST("MNIST_data/",download=True,train=True,transform=transform)
trainloader=torch.utils.data.DataLoader(trainset,batch_size=64,shuffle=True)

dataiter=iter(trainloader)
images,labels=dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)
plt.imshow(images[1].numpy().squeeze())

inputs=images.view(images.shape[0],-1)    #flattened 28*28 =784
print (inputs.shape)
print (images.shape)

def activation(x):
    return 1/(1+torch.exp(-x))


def softmax(x):
    xx=torch.exp(x)/torch.sum(torch.exp(x),dim=1).view(-1,1)
    return xx

# neural networks
w1=torch.randn(784,256)
b1=torch.randn(256)

w2=torch.randn(256,10)
b2=torch.randn(10)
h=activation(torch.mm(inputs,w1)+b1)
out=torch.mm(h,w2)+b2
out1=softmax(out)


from torch import nn
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(784,256)
        self.output=nn.Linear(256,10)
        self.sigmoid=nn.Sigmoid()
        self.softmax=nn.Softmax(dim=1)
    
    def forward(self,x):
        x=self.hidden(x)
        x=self.sigmoid(x)
        x=self.output(x)
        x=self.softmax(x)
        return x 
 

import torch.nn.functional as F
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden=nn.Linear(784,256)
        self.output=nn.Linear(256,10)

    
    def forward(self,x):
        x=F.sigmoid(self.hidden(x))
        x=F.softmax(self.hidden(x))
        return x 
 

model=Network()
    
    
    