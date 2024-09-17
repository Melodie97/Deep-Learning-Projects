import json
import torch
import models as md
import helper
import torch.nn.functional as F
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models


def main():
    # Get input arguments
    args = helper.get_train_input_args()
    
    # Set the device
    device = torch.device('cuda' if args.gpu and torch.cuda.is_available() else 'cpu')
    
    # Load data
    train_loader, valid_loader, class_to_idx = helper.load_data(args.data_dir)
    
    # Build model
    model = md.build_model(arch=args.arch, hidden_units=args.hidden_units, output_size=len(class_to_idx))
    model.to(device)
    
    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    if args.arch.startswith('resnet'):
        optimizer = optim.Adam(model.fc.parameters(), lr=args.learning_rate)
    else:
        optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)
    
    # Train the model
    md.train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=2)
    
    # Save the model checkpoint
    md.save_checkpoint(model, args.save_dir, args.arch, class_to_idx, args.hidden_units, optimizer, args.epochs)

if __name__ == '__main__':
    main()