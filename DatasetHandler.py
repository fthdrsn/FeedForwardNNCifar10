
### Import required packages ####
import torch
import torchvision  #To load dataset
import torchvision.transforms as transforms #To apply transformations on dataset
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np

## To transfer samples to selected device
class ToDevice():
    def __init__(self,device,dataLoader):
        self.device=device
        self.dataLoader=dataLoader
    def __iter__(self):
        for data,label in self.dataLoader:
            yield [data.to(self.device),label.to(self.device)]

class DatasetManager():

    def __init__(self,args):
        self.datasetPath=args['datasetPath']
        self.normalizeData=args['normalizeData']
        self.validationPercent=args['validationPercent']
        self.augmentData=args['augmentData']
        self.shuffleData=args['shuffleData']
        self.batchSize=args['batchSize']
        self.batchSizeFrw = args['batchSizeFrw']
        self.device=args['device']

        self.trainData=None
        self.testData=None
        self.trainDataLoader=None
        self.testDataLoader=None
        self.valDataLoader=None

        self.nValSamples = None
        self.nOnlyTrainSamples = None
        self.nTestSamples =None

    ##Download dataset if it is not already exist.
    def __call__(self):
        transform=[]
        if self.augmentData:
            transform+=[transforms.RandomCrop(32, padding=4),transforms.RandomHorizontalFlip()]

        transform+=[transforms.ToTensor()]

        if self.normalizeData:
            transform+=[transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))]

        transform=transforms.Compose(transform)

        self.trainData= torchvision.datasets.CIFAR10(root=self.datasetPath, train=True,download=True, transform=transform)
        self.testData = torchvision.datasets.CIFAR10(root=self.datasetPath, train=False, download=True, transform=transform)

        if self.trainData is None or self.testData is None:
            raise Exception("Dataset is not found")

    def GetDatasets(self):
        if self.trainData is None or self.testData is None:
            raise Exception('Please run call function first')
        return self.trainData.data,self.testData.data

    def GetDataLoaders(self):
        if self.trainData is None or self.testData is None:
            raise Exception('Please run call function first')

        nAllTrainSamples = len(self.trainData)  # Number of samples in train set
        self.nValSamples = int(np.floor(nAllTrainSamples * self.validationPercent))  # Number of validation samples
        self.nOnlyTrainSamples=nAllTrainSamples-self.nValSamples
        self.nTestSamples=len(self.testData)
        indexes = list(range(nAllTrainSamples))  # Sample indexes in train set
        np.random.shuffle(indexes)  # Shuffle indices

        valIndexes = indexes[:self.nValSamples]  # Get validation indices
        trainIndexes = indexes[self.nValSamples:]  # Get train indices

        # Create data samplers
        trainSampler = SubsetRandomSampler(trainIndexes)
        validSampler = SubsetRandomSampler(valIndexes)
        # Create data loaders
        self.trainDataLoader = torch.utils.data.DataLoader(self.trainData, batch_size=self.batchSize, sampler=trainSampler, num_workers=2)
        self.valDataLoader = torch.utils.data.DataLoader(self.trainData, batch_size=self.batchSizeFrw, sampler=validSampler, num_workers=2)
        self.testDataLoader = torch.utils.data.DataLoader(self.testData, batch_size=self.batchSizeFrw, shuffle=False, num_workers=2)

        self.trainDataLoader = ToDevice(self.device,self.trainDataLoader)
        self.valDataLoader = ToDevice(self.device,self.valDataLoader)
        self.testDataLoader = ToDevice(self.device,self.testDataLoader)

        return self.trainDataLoader,self.valDataLoader,self.testDataLoader

    def GetImageShape(self):
        if self.trainData is None or self.testData is None:
            raise Exception('Please run call function first')
        return self.trainData.data[0].shape

    def GetDataClasses(self):
        if self.trainData is None or self.testData is None:
            raise Exception('Please run call function first')
        return self.trainData.classes

    def GetNumOfSamples(self):
        if self.trainData is None or self.testData is None:
            raise Exception('Please run call function first')
        return self.nOnlyTrainSamples,self.nValSamples,self.nTestSamples



