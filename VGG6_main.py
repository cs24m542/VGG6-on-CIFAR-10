import argparse
import json
import torch
import torch.nn as nn

from image_transform import GetCifar10  
from model import cfg_vgg6, VGG, make_layers
from train import train_model
from utils import eval_model,set_seed 

import warnings
from pydantic.warnings import PydanticDeprecatedSince20, UnsupportedFieldAttributeWarning

warnings.filterwarnings("ignore", category=UnsupportedFieldAttributeWarning)
warnings.filterwarnings("ignore", category=PydanticDeprecatedSince20)

#Try to import wandb safely
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--activation', type=str, default='ReLU')
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--learning_rate', type=float, default=0.01)
    p.add_argument('--optimizer', type=str, default='SGD')
    p.add_argument('--seed', type=int, default=14)
    p.add_argument('--project', type=str, default='vgg6_cifar10')
    p.add_argument('--wandb_mode', type=str,default='Disabled')
    p.add_argument('--mode', type=str, default="train",choices=['train', 'val'])
    p.add_argument('--val_mode', type=str, default="False")
    #p.add_argument('--val_mode', action='store_true', help="Enable validation mode")
    p.add_argument('--weight_file_path', type=str, default='./weights.pth')
    a = p.parse_args()
    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )
    set_seed(a.seed)

    config_dict = {
        "activation": a.activation,
        "batch_size": a.batch_size,
        "epochs": a.epochs,
        "learning_rate": a.learning_rate,
        "optimizer": a.optimizer
    }
    
    if WANDB_AVAILABLE:
        if a.wandb_mode=="wandb_standalone" :
            wandb.init(project=a.project, config=config_dict)
            config = wandb.config
            
        elif a.wandb_mode=="wandb_sweep":
            # Running under a sweep agent — already initialized
            wandb.init(project=a.project)
            config = wandb.config
            # Normal W&B run      
        else:
            # No wandb logging
            config = config_dict
            wandb=None
    else:
        config = config_dict
        wandb=None


    print(config)

    activation_map = {
        "ReLU": nn.ReLU,
        "Sigmoid": nn.Sigmoid,
        "Tanh": nn.Tanh,
        "SiLU": nn.SiLU,
        "GELU": nn.GELU
    }

    train_loader, val_loader, test_loader = GetCifar10(config["batch_size"])
    model = VGG(make_layers(cfg_vgg6, activation=activation_map[config["activation"]])).to(device)
    criterion = nn.CrossEntropyLoss()
    if a.mode=="train":
        train_model(model, config, train_loader, val_loader,test_loader,criterion=criterion, wandb=wandb, device=device)
        torch.save(model.state_dict(), a.weight_file_path)
        print(f"Model weights saved to {a.weight_file_path}")

    elif a.mode=="val":
        model.load_state_dict(torch.load(a.weight_file_path, map_location=device))
        print(f"Loaded model weights from {a.weight_file_path}")
        val_acc, val_loss = eval_model(model, val_loader, criterion, device)
        print(f"Validation Accuracy: {val_acc:.2f}%, Loss: {val_loss:.4f}")
        Test_acc, Test_loss = eval_model(model, test_loader, criterion, device)
        print(f"Test Accuracy: {Test_acc:.2f}%, Loss: {Test_loss:.4f}")

    else:
        print("Please specify either --train_mode==True or --val_mode==True")
