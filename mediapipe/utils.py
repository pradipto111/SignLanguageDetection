import torch, cv2
from torch.utils.data import Dataset
from tqdm.auto import tqdm
import torch.nn.functional as F
import glob

classes = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'del', 'nothing', 'space']

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


class LandmarksDataset(Dataset):
    """Reads the mediapipe landmarks from the directory, 
    equivalent to the ImageFolder class meant for images.
    """
    def __init__(self, root):
        self.path = root
        file_list = glob.glob(self.path + "/*")
        self.data = []
        for class_path in file_list:
            class_name = class_path.split("/")[-1]
            for path in glob.glob(class_path + "/*.pt"):
                self.data.append([path, class_name])
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        path, class_name = self.data[idx]
        lms = torch.load(path, 'cpu')
        lms = lms - lms[0]
        lms = lms[1:, :]
        class_id = classes.index(class_name)
        return lms, class_id


def generate_landmarks(img, hands):
  """
  Processes a hand image, and generates the landmarks, and transfers origin
  """
  imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  results = hands.process(imgRGB)
  lms = torch.zeros(21,2)
  if results.multi_hand_landmarks:
    for handLms in results.multi_hand_landmarks:
      for i in range(21):
        lms[i][0] = handLms.landmark[i].x
        lms[i][1] = handLms.landmark[i].y
  #lms = to_device(lms, device)
  lms = lms - lms[0]
  lms = lms[1:]
  return lms

