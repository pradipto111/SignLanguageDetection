#Importing libraries
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from tqdm.auto import tqdm
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from model import ASLMobilenet, ASLResnet
import argparse
import warnings
from utils import *

# Definition of arguments for ease of access and training customization
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str, help = "Path to the dataset")
parser.add_argument("model", type=str, choices=['resnet34', 'mobilenet_v2'],help = "Model to be used for training, 'resnet34' OR 'mobilenet_v2")
parser.add_argument("--epochs", type=int, default=10, help = "Number of epochs for training")
parser.add_argument("--learning_rate", type=float, default = 1e-5, help = "Maximum Learning Rate")
parser.add_argument("--weight_decay", type=float, default =  1e-4, help = "Weight Decay for learning rate scheduling")
parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient Clipping Coefficient")
parser.add_argument("--batch_size", type=int, default=32, help = "Batch Size for training and validation")




# main begins here
if __name__== "__main__":
    args = parser.parse_args() #parse the arguments for training customisation


    #Save the argument values##########
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    GRAD_CLIP = args.grad_clip
    DATASET_PATH = args.dataset_path
    EPOCHS = args.epochs
    MODEL = None
    ###################################


    ##set the model##############
    if args.model == 'resnet34':
        MODEL = ASLResnet
    else:
        MODEL = ASLMobilenet
    #############################

    warnings.filterwarnings("ignore")
    device = get_default_device() #get default device (cpu or cuda(gpu))


    ##Loading the desired dataset #####################################
    print('Loading Dataset ... ')
    valid_ds = ImageFolder(DATASET_PATH+'/valid', transform = ToTensor()) # Validation dataset
    train_ds = ImageFolder(DATASET_PATH+'/train', transform = ToTensor()) # Training dataset
    print('Dataset Loaded.')
    ###################################################################


    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE, pin_memory=True) #Create the train dataloader that feeds thh data into the model
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True) #Create the validation dataloader that feeds the data into the model

    train_dl = DeviceDataLoader(train_dl, device) #Load the train dataloaders onto the device (cpu or cuda)
    valid_dl = DeviceDataLoader(valid_dl, device) #Load the validation dataloaders onto the device (cpu or cuda)

    model = to_device(MODEL(), device) # Load the model onto the device (cpu or cuda)

    opt_func = torch.optim.Adam #set the optimizer function as ADAM

    history = fit_one_cycle(EPOCHS, LEARNING_RATE, model, train_dl, valid_dl, 
                             grad_clip=GRAD_CLIP, 
                             weight_decay=WEIGHT_DECAY, 
                             opt_func=opt_func) #performs the training and returns a 'history' list which containes details of the whole run
                             
    save_model(model) #save the model