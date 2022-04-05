#Import necessary libraries
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from tqdm.auto import tqdm
from model import SignLanguageModel
import warnings
import argparse
from utils import *
warnings.filterwarnings("ignore")

#Class labels
classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

#Define the arguements for ease of access and customizable training sessions
parser = argparse.ArgumentParser()
parser.add_argument("dataset_path", type=str, help = "Path to the dataset")
parser.add_argument("--epochs", type=int, default=10, help = "Number of epochs for training")
parser.add_argument("--learning_rate", type=float, default = 1e-5, help = "Maximum Learning Rate")
parser.add_argument("--weight_decay", type=float, default =  1e-4, help = "Weight Decay for learning rate scheduling")
parser.add_argument("--grad_clip", type=float, default=0.1, help="Gradient Clipping Coefficient")
parser.add_argument("--batch_size", type=int, default=32, help = "Batch Size for training and validation")


#Main begins here
if __name__== "__main__":

    args = parser.parse_args() #parse the arguments for training customisation

    #Save the argument values##########
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.learning_rate
    WEIGHT_DECAY = args.weight_decay
    GRAD_CLIP = args.grad_clip
    DATASET_PATH = args.dataset_path
    EPOCHS = args.epochs
    ###################################


    
    MODEL = SignLanguageModel #set the model


    device = get_default_device() #get default device (cpu or cuda(gpu))

     ##Loading the desired dataset #####################################
    print('Loading Dataset ... ') 
    valid_ds = LandmarksDataset(DATASET_PATH + '/valid')
    train_ds = LandmarksDataset(DATASET_PATH + '/train')
    print('Dataset Loaded.')
    ###################################################################


    train_dl = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=BATCH_SIZE, pin_memory=True) #Create the train dataloader that feeds thh data into the model
    valid_dl = DataLoader(valid_ds, BATCH_SIZE, num_workers=BATCH_SIZE, pin_memory=True) #Create the validation dataloader that feeds the data into the model

    train_dl = DeviceDataLoader(train_dl, device) #Load the train dataloaders onto the device (cpu or cuda)
    valid_dl = DeviceDataLoader(valid_dl, device) #Load the validation dataloaders onto the device (cpu or cuda)

    model = to_device(MODEL(), device) # Load the model onto the device (cpu or cuda)

    opt_func = torch.optim.Adam # set the optimizer as ADAM

    history = fit_one_cycle(EPOCHS, LEARNING_RATE, model, train_dl, valid_dl, 
                             grad_clip=GRAD_CLIP, 
                             weight_decay=WEIGHT_DECAY, 
                             opt_func=opt_func) #performs the training and returns a 'history' list which containes details of the whole run
    
    save_model(model) #save the model