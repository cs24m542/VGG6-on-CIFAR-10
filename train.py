from PIL import Image, ImageEnhance, ImageOps
import random
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import random_split
#from cutout_autoaugment import Cutout, CIFAR10Policy  # your augmentation classes
import itertools
from model import cfg_vgg6,VGG,make_layers
from utils import eval_model 


# -------------------------------
# Training Function
# -------------------------------
def train_model(model,config,train_dataloader=None,val_dataloader=None,test_dataloader=None,criterion=nn.CrossEntropyLoss(),wandb=None,device=None):
    best_acc = 0
    
    #early_stopping = EarlyStopping(patience=5, min_delta=0.001)

    # Read hyperparameters from config
    activation_map = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "SiLU": nn.SiLU,
        "GELU": nn.GELU
    }
    #train_loader,val_loader,test_loader = GetCifar10(config["batch_size"])
    
    #model = VGG(make_layers(cfg_vgg6, activation=activation_map[config["activation"]])).to(device)
    print(model)
    #criterion = nn.CrossEntropyLoss()
    
     # Optimizer selection
    if config["optimizer"] == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9)
    elif config["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "Adagrad":
        optimizer = optim.Adagrad(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == "RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'Nadam':
        optimizer = optim.NAdam(model.parameters(), lr=config["learning_rate"])
    elif config["optimizer"] == 'Nesterov':
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"], momentum=0.9,nesterov=True)
    else:
        raise ValueError("Unknown optimizer")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    best_acc = 0

    
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        for data, target in train_dataloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            running_loss += loss.item()
        
        if scheduler:
            scheduler.step()
        # Evaluate every  epochs
        #if (epoch + 1) % 5 == 0 or epoch == 0:
        if True:
            train_acc,train_loss = eval_model(model, train_dataloader,criterion,device)
            val_acc,val_loss = eval_model(model,val_dataloader,criterion,device)
            test_acc,test_loss = eval_model(model,test_dataloader,criterion,device)
            print(f"Epoch {epoch+1:03d} - Train Loss: {running_loss/len(train_dataloader):.4f} val Loss: {val_loss:.4f}, Test Loss: {test_loss:.4f} "
                  f"Train Acc: {train_acc:.2f}%, val Acc: {val_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
            if(wandb is not None):
                wandb.log({
                    "epoch": epoch + 1,
                    "train_accuracy": train_acc,
                    "val_accuracy": val_acc,
                    "test_accuracy": test_acc,
                    "train_loss": running_loss/len(train_dataloader),
                    "val_loss": val_loss,
                    "test_loss": test_loss
                })
            if val_acc > best_acc:
                best_acc = val_acc
            #early_stopping(val_loss)
            #if early_stopping.early_stop:
            #    print(f"Early stopping at epoch {epoch+1}")
            #    break
            #    torch.save(model.state_dict(), "best_vgg6_cifar10.pth")
            if(wandb is not None):
                wandb.log({"best_test_acc": best_acc})
            
    if(wandb is not None):
        wandb.log({"best_val_acc": best_acc})
    print(f"Best Accuracy: {best_acc:.2f}%")
