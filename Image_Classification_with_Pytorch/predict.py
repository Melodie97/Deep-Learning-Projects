import os
import json
import torch
import numpy as np
import models as md
import helper
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from PIL import Image

# TODO: Write a function that loads a checkpoint and rebuilds the model

def main():
    print(torch.__version__)
    
    args = helper.get_predict_input_args()
    
    # Check for GPU usage
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Load the checkpoint
    model = md.load_checkpoint(args.checkpoint)
    model.to(device)
    
    # Load category names
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    
    # Predict the class
    probs, classes = md.predict(args.image_path, model, args.top_k, device)
    
    # Map classes to names
    names = [cat_to_name[cls] for cls in classes]
    
    # Print results
    for prob, name in zip(probs, names):
        print(f"{name}: {prob*100:.2f}%")
    
if __name__ == '__main__':
    main()