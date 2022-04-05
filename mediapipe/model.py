import torch.nn as nn
import torch.nn.functional as F
import torch

class Classification(nn.Module):
    """
    Akash Rao NS
    Image Classification using Convolutional Neural Networks in PyTorch
    https://jovian.ai/aakashns/05-cifar10-cnn

    This class manages the computations of the model behind the curtains and records the data for every batch and epoch. 
    It also takes care of diaplying the necessary details to the developer after every training epoch.
    """
    def training_step(self, batch):
      # passes an tensor batch forward to the model layers, and returns the cross entropy loss
        images,labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss
    
    def validation_step(self,batch):
      # calculates loss and accuracy for a validation batch and returns these values as a dictionary
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = self.accuracy(out, labels)
        return {'val_loss': loss.detach(), 'val_acc': acc}
    
    def validation_epoch_end(self, outputs):
    # after 1 epoch, calculates the mean validation loss and accuracy for all batches and returns these values as a dictionary
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
       # prints out information related to an epoch at the end of it
        print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
            epoch+1, result['train_loss'], result['val_loss'], result['val_acc']))
        
    def accuracy(self,outputs,labels):
       # calculates accuracy of the outs with respect to labels, fed into the function as arguments
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))


class SignLanguageBlock(nn.Module):
  """
  A template forward block to be used in the main model architecture.
  Customizable input and output channels, initialised as user defined arguments.
  """
  def __init__(self, in_ch, out_ch):
    super(SignLanguageBlock, self).__init__()
    self.layer = nn.Sequential(
        nn.Linear(in_ch, 1024),
        nn.BatchNorm1d(1024),
        nn.ReLU(),
        nn.Linear(1024,128),
        nn.Tanh(),
        nn.Linear(128, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Linear(128,out_ch),
        nn.Tanh()
    )    

  def forward(self, x):
    x = self.layer(x)
    return x

class SignLanguageModel(Classification):
  """
  Definition of the model layers.
  """
  def __init__(self):
    super().__init__()
    self.network = nn.Sequential(
        nn.Flatten(),
        SignLanguageBlock(40,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,1024),
        SignLanguageBlock(1024,512),
        SignLanguageBlock(512,128),
        SignLanguageBlock(128,8),
        nn.Linear(8, 29),
        nn.Softmax()
    )
  def forward(self, x):
    return self.network(x)