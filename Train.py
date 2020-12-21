import argparse
import torch
import numpy as np
from DatasetHandler import DatasetManager
from NNModels import FeedForwardNetwork
import time
from scipy.io import savemat
import random
from torch.utils.tensorboard import SummaryWriter
import os

## Save trained model
def SaveModel(model,savePath):
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    path = savePath +"/bestmodel.dat"
    torch.save(model.state_dict(), path)

## Load trained model
def LoadModel(model,loadPath,device):
    try:
        model.load_state_dict(torch.load(loadPath, map_location=device))
    except FileNotFoundError:
        print("model not found")

def Start(args):
    #Fixed seed for reproducibility
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])

    #Select gpu as device if exist
    if args["device"]=='cuda':
        args["device"]='cuda' if torch.cuda.is_available() else 'cpu'
    else:
        args["device"]='cpu'

    #Save hyperparameters to tensorboard
    if args["saveResults"]:
        writer = SummaryWriter(comment="_[ALL_New100Epoch]")
        writer.add_hparams({"LearningRate": args["learningRate"],"BatchSize":args["batchSize"],"DropOut":args["applyDropOut"],"HiddenDim":args["hiddenDim"],
                            "Augment":args["augmentData"],"ManualWeightInit":args["manualWeightInit"],"Regularization":args["useRegularization"],"Normalize:":args["normalizeData"]},
                           {"_": 0})

    #Create dataset manager object
    datasets=DatasetManager(args)
    datasets() #Run call function to download datasets and apply transformations
    shapeOfData = datasets.GetImageShape() ##Get shape of image in dataset
    classes=datasets.GetDataClasses() ##Classes of datasets
    trainLoader,valLoader,testLoader=datasets.GetDataLoaders() ##Get dataloader for training,validation and test datasets.
    nTrainSamples, nValidSamples, nTestSamples = datasets.GetNumOfSamples() ##Get #samples in each dataset

    inputDim = shapeOfData[0] * shapeOfData[1] * shapeOfData[2]  ##Input shape for Feed Forward NN
    outputDim=len(classes)
    hiddenDim=args['hiddenDim']
    nnModel=FeedForwardNetwork(inputDim,hiddenDim,outputDim,args).to(args["device"])

    ## Select optimizer(Use L2 regularization if specified)
    if args['useRegularization']:
        optimizer=torch.optim.Adam(nnModel.parameters(),args['learningRate'],weight_decay=args["regRate"])
    else:
        optimizer = torch.optim.Adam(nnModel.parameters(), args['learningRate'])

    #Define loss function
    lossFunc=torch.nn.CrossEntropyLoss()
    #Define accuracy function
    accFunc=lambda predClass,labels: torch.tensor(torch.sum(predClass == labels).item() / len(predClass))

    TrainLoss=[] #Store training loss for all epochs.
    TrainAcc = [] #Store training accuracy for all epochs.
    ValLoss = [] #Store validation loss for all epochs.
    ValAcc = [] #Store validation accuracy for all epochs.
    earlyStopVar=0
    if not args["runMode"]=="Test":
        currentTime = time.time()
        for e in range(1,args['numOfEpoch']+1):

            ############# TRAINING PHASE ####################
            ## Epoch dependant variables.
            epochTrainAcc=0
            epochTrainLoss = 0
            epochValAcc = 0
            epochValLoss = 0
            for data,labels in trainLoader:

                pred=nnModel(data) #Predictions in shape (batchsize,#classes)
                loss=lossFunc(pred,labels)
                predClass = torch.argmax(pred.detach(), dim=1)  # Predicted class
                acc = accFunc(predClass, labels)
                epochTrainAcc += acc
                epochTrainLoss+=loss.detach()

                optimizer.zero_grad() #Clear gradients
                loss.backward() #Calculate gradients
                optimizer.step() #Propogate gradients to network.

            epochTrainLoss=epochTrainLoss/nTrainSamples * args['batchSize']
            epochTrainAcc=epochTrainAcc/nTrainSamples * args['batchSize']
            TrainLoss.append(epochTrainLoss)
            TrainAcc.append(epochTrainAcc)
            print(f"Epoch: {e}  Training Loss: {epochTrainLoss}  Training Accuracy: {epochTrainAcc}")

            ############# VALIDATION PHASE ####################
            # Do not apply validation step if greedy search option is selected.
            if not args["runMode"]=="GreedySearch":
                with torch.no_grad(): #Stop calculating gradients in validation phase(Just forward pass)
                    nnModel.eval() #Switch model to evaluation mode not to apply batch normalization and dropouts in forward pass
                    for dataVal, labelsVal in valLoader:
                        predVal = nnModel(dataVal)  # Predictions in shape (batchsize,#classes)
                        lossVal = lossFunc(predVal, labelsVal)
                        predClassVal = torch.argmax(predVal, dim=1)  # Predicted class
                        accVal = accFunc(predClassVal, labelsVal)
                        epochValAcc += accVal
                        epochValLoss += lossVal
                    nnModel.train() #Switch model back to training mode

                epochValLoss = epochValLoss / nValidSamples * args['batchSizeFrw']
                epochValAcc = epochValAcc / nValidSamples * args['batchSizeFrw']
                print(f"Epoch: {e}  Validation Loss: {epochValLoss}  Validation Accuracy: {epochValAcc}")

                ## Save results to tensorboard
                if args["saveResults"]:
                    writer.add_scalar("TrainingLoss", epochTrainLoss, e)
                    writer.add_scalar("TrainingAcc" , epochTrainAcc, e)
                    writer.add_scalar("ValidationLoss", epochValLoss, e)
                    writer.add_scalar("ValidationAcc", epochValAcc, e)

                ValLoss.append(epochValLoss)
                ValAcc.append(epochValAcc)

                if args["earlyStopping"]:
                    if epochValLoss>np.min(ValLoss):
                        if earlyStopVar==0:
                            checkPoint=nnModel.parameters()
                        earlyStopVar+=1
                    else:
                        earlyStopVar=0

                    if earlyStopVar>=args["earlyStopEpoch"]:
                        print("#####################################")
                        print("#############EARLY STOPPING##########")
                        print(f"Training early stopped in {e}.epoch for Validation Loss: {epochValLoss} and Validation Accuracy: {epochValAcc}")
                        print(f"Returned {e-args['earlyStopEpoch']}.epoch with"
                              f" Validation Loss: {ValLoss[e-args['earlyStopEpoch']-1]} and Validation Accuracy: {ValAcc[e-args['earlyStopEpoch']-1]}")
                        print("#####################################")

                        for p1,p2 in zip(checkPoint,nnModel.parameters()):
                            p2.data.copy_(p1.data)
                        break

        passTime=time.time()-currentTime

    if args["runMode"]=="Test" or args["runMode"]=="TrainTest":
        print("#####################################")
        print("#############TEST PHASE##############")

        if args["runMode"]=="Test":
            LoadModel(nnModel,"BestModel/bestmodel.dat",args["device"])

        epochTestAcc=0
        epochTestLoss=0
        nnModel.eval()
        for dataTest, labelsTest in testLoader:
            # Loss and accuracy for test set
            predTest= nnModel(dataTest)  # Predictions in shape (batchsize,#classes)
            lossTest= lossFunc(predTest, labelsTest)
            predClassTest = torch.argmax(predTest, dim=1)  # Predicted class
            accTest = accFunc(predClassTest, labelsTest)
            epochTestAcc += accTest
            epochTestLoss += lossTest

        epochTestLoss =  epochTestLoss / nTestSamples * args['batchSizeFrw']
        epochTestAcc =  epochTestAcc / nTestSamples * args['batchSizeFrw']

        print(f"Test Loss: {epochTestLoss }  Test Accuracy: {epochTestAcc}")
        print("#####################################")

    if args["saveModel"]:
        SaveModel(nnModel, "BestModel")

    if args["runMode"]=="GreedySearch":
        return epochTrainAcc,epochTrainLoss,passTime

def str2bool(value):
    return value == "True"

if __name__=="__main__":
    argsParser = argparse.ArgumentParser(description='Fatih Dursun-Feed Forward Neural Network for CIFAR10 Dataset')

    argsParser.add_argument("--device", default='cuda', help="device to work on")
    ##Dataset related arguments
    argsParser.add_argument("--datasetPath", default="./data", help="the location of dataset to be downloaded")
    argsParser.add_argument("--validationPercent", type=float,default=0.1, help="percentage of the validation samples w.r.t all training samples")
    argsParser.add_argument("--shuffleData", type=str2bool, default=True, help="shuffle the training dataset if true")
    argsParser.add_argument("--batchSize", type=int, default=128, help="batch size for training")
    argsParser.add_argument("--batchSizeFrw", type=int, default=128, help="batch size for validation and test phases")

    #Model related arguments
    argsParser.add_argument("--hiddenDim", type=int, default=92,help="hidden size of NN")
    argsParser.add_argument("--learningRate", type=float, default=0.001, help="learning rate of NN")

    #Training loop
    argsParser.add_argument("--numOfEpoch", type=int, default=40, help="number of training epoch")
    argsParser.add_argument("--seed", type=int, default=0, help="fixed seed for reproducibility")
    argsParser.add_argument("--dropoutRate", type=float, default=0.2, help="Dropout rate")
    argsParser.add_argument("--regRate", type=float, default=0.001, help="Regularization rate")
    argsParser.add_argument("--saveResults", type=str2bool, default=False, help="If true, save results to tensorboard")
    argsParser.add_argument("--saveModel", type=str2bool, default=False, help="If true, save model")
    argsParser.add_argument("--earlyStopEpoch", type=int, default=3, help="If the model's validation loss doesn't decrease for consecutive specified number of epochs, stop training earlier")
    argsParser.add_argument("--runMode", default="TrainTest", help="The running mode of app. Can be set four option:"
                                                                   "TrainTest: Train and test model" 
                                                                   "Train: Just train and validate,do not test"
                                                                   "Test: Just test model (an already trained model should be saved before running this mode)"
                                                                   "GreedySearch: Find best batchsize vs learning rate values(Do not run validation and test)")

    ##Switchs to toggle model improvements
    argsParser.add_argument("--augmentData", type=str2bool, default=False, help="augment train set if true")
    argsParser.add_argument("--manualWeightInit",  type=str2bool, default=False, help="if true, manually initialize weights")
    argsParser.add_argument("--useRegularization", type=str2bool, default=False, help="if true, activate l2 regularisation")
    argsParser.add_argument("--normalizeData", type=str2bool, default=True, help="determines whether normalize the dataset between [-1,1]")
    argsParser.add_argument("--applyDropOut", type=str2bool, default=False,help="if true, apply dropout")
    argsParser.add_argument("--batchNorm", type=str2bool, default=False, help="if true, apply batch normalization")
    argsParser.add_argument("--earlyStopping", type=str2bool, default=False, help="if true, activate early stopping")

    args = vars(argsParser.parse_args())

    if args['runMode']=="GreedySearch":
        batchList = [8, 16, 32, 64, 128]
        lrList = [0.001, 0.005, 0.01, 0.05, 0.1]
        bCount=len(batchList)
        lCount=len(lrList)
        AccBatchVsLr=np.zeros((bCount,lrList))
        LossBatchVsLr = np.zeros((bCount,lrList))
        TimeBatchVsLr = np.zeros((bCount,lrList))
        for i,b in enumerate(batchList):
            for j,l in enumerate(lrList):
                args['batchSize']=b
                args['learningRate'] = l
                acc,loss,passedTime=Start(args)
                AccBatchVsLr[i,j]=acc
                LossBatchVsLr[i, j] = loss
                TimeBatchVsLr[i, j] = passedTime
                print(f"Batch{b} Lr{l} Acc{acc} Loss{loss} ")

        mdicAcc = {"Accuracy": AccBatchVsLr, "label": "experimentAccuracy"}
        mdicLoss = {"Loss": LossBatchVsLr, "label": "experimentLoss"}
        mdicTime = {"Time": TimeBatchVsLr, "label": "experimentTime"}
        savemat("Accuracy.mat", mdicAcc)
        savemat("Loss.mat", mdicLoss)
        savemat("Time.mat", mdicTime)
    else:
        Start(args)

