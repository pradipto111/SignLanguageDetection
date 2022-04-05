# Import necessary libraries
import torch
from tqdm.auto import tqdm
import torch.nn.functional as F
import torch.nn as nn


def get_default_device():
    """
    Pick GPU if available, else CPU
    Akash Rao NS
    Image Classification using Convolutional Neural Networks in PyTorch
    https://jovian.ai/aakashns/05-cifar10-cnn
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """
    Move tensor(s) to chosen device
    Akash Rao NS
    Image Classification using Convolutional Neural Networks in PyTorch
    https://jovian.ai/aakashns/05-cifar10-cnn
    """
    if isinstance(data, (list,tuple)):

        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    """
    Wrap a dataloader to move data to a device
    Akash Rao NS
    Image Classification using Convolutional Neural Networks in PyTorch
    https://jovian.ai/aakashns/05-cifar10-cnn
    """
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
        
    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl: 
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def evaluate(model, val_loader):
    # performs evluation for validation batches, given the validation dataloader
    # and returns average loss and accuracy for the epocj
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    # Returns current learning rate value for the optimizer
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader, 
                  weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    # the core function that runs the training epochs
    torch.cuda.empty_cache()
    history = []
    
    # Set up custom optimizer with weight decay
    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    # Set up one-cycle learning rate scheduler
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs, 
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(epochs):
        # Training Phase 
        model.train()
        train_losses = []
        lrs = []
        print('Epoch:', epoch+1)
        for batch in tqdm(train_loader):
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()
            
            # Gradient clipping
            if grad_clip: 
                torch.nn.utils.clip_grad_value_(model.parameters(), grad_clip)
            
            optimizer.step()
            optimizer.zero_grad()
            
            # Record & update learning rate
            lrs.append(get_lr(optimizer))
            sched.step()
        
        # Validation phase
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        model.epoch_end(epoch, result)
        history.append(result)
    return history


def save_model(model):
    """Saves the model at the end of training"""
    ans = input('Save model (y/n)?')
    if ans == 'y' or ans == 'Y':
        name = input('Model Name: ')
        version = input('Model Version: ')
        torch.save(model.state_dict(), 'models/model_' + name + '_' + version + '.pt')
    elif ans == 'n' or ans == 'N':
        return
    else:
        print('Enter a valid response.')
        save_model(model)


def mask_image(img):
    """
    This function takes an image array as input and performs skin segmentation on it based on HSV and RGB thresholds.
    """
    imgY = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mask1 = np.logical_and(np.logical_and((img[:, :, 0] > 95),(img[:, :, 1] > 40)),(img[:, :, 2] > 20))*1
    mask2 = (np.abs(img[:, :, 0] - img[:, :, 1]) >  15)*1
    mask3 = np.logical_and((img[:, :, 0] > img[:, :, 1]),(img[:, :, 0] > img[:, :, 2]))*1
    mask4 = imgHSV[:,:,0]<=50
    mask5 = np.logical_and(imgHSV[:, :, 1] <= 0.68, imgHSV[:,:, 1]>=0.23)
    RGBmask = (mask1 * mask2 * mask3 * mask4)*255
    RGBmask = RGBmask.astype(np.uint8)
    return cv2.bitwise_and(img, img, mask = RGBmask)



def create_confusion_matrix(y_true, y_pred, classes):
    """ creates and plots a confusion matrix given two list (targets and predictions)
    :param list y_true: list of all targets (in this case integers bc. they are indices)
    :param list y_pred: list of all predictions (in this case one-hot encoded)
    :param dict classes: a dictionary of the countries with they index representation
    """
    classes = {}
    for l in labels:
      classes[l] = labels.index(l)
    amount_classes = len(classes)

    confusion_matrix = np.zeros((amount_classes, amount_classes))
    for idx in range(len(y_true)):
        target = y_true[idx]

        output = y_pred[idx]

        confusion_matrix[target][output] += 1

    plt.figure(figsize = (5,5))
    fig, ax = plt.subplots(1)

    ax.matshow(confusion_matrix)
    ax.set_xticks(np.arange(len(list(classes.keys()))))
    ax.set_yticks(np.arange(len(list(classes.keys()))))

    ax.set_xticklabels(list(classes.keys()))
    ax.set_yticklabels(list(classes.keys()))

    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.show()