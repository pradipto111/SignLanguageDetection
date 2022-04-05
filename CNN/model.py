import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision.models as models

class Classification(nn.Module):
    """
    Akash Rao NS
    Image Classification using Convolutional Neural Networks in PyTorch
    https://jovian.ai/aakashns/05-cifar10-cnn

    This class manages the computations of the model behind the curtains and records the data for every batch and epoch. 
    It also takes care of diaplying the necessary details to the developer after every training epoch.
    """
    def training_step(self, batch):
        # passes an image batch forward to the CNN layers, and returns the cross entropy loss
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


class ASLResnet(Classification):
    """
    Gryan Galario
    https://jovian.ai/gry-galario/project/v/5?utm_source=embed

    This is the model definition for ASL Resnet
    """
    def __init__(self):
        super().__init__()
        self.network = models.resnet34(pretrained=True)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 24)
    
    def forward(self, xb):
        return self.network(xb)
    
    def freeze(self):
        for param in self.network.parameters():
            param.require_grad = False
        for param in self.network.fc.parameters():
            param.require_grad = True
    
    def unfreeze(self):
        for param in self.network.parameters():
            param.require_grad = True

class ASLMobilenet(Classification):
    """
    This is the model definition for ASL MobilenetV2
    """
    def __init__(self):
        super().__init__()
        self.network = models.mobilenet_v2(pretrained = True)
        self.network.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.network.last_channel, 24),
        )
    
    def forward(self, xb):
        return self.network(xb)
