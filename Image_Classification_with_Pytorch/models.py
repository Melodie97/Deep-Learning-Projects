import os
import argparse
import torch
import helper
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from collections import OrderedDict

def build_model(arch='densenet121', hidden_units=512, output_size=102):
    """Build and return the model based on the specified architecture."""
    # Load a pretrained network
    model = getattr(models, arch)(pretrained=True)
    
    # Freeze parameters to avoid backpropagation through them
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier layer
    if arch.startswith('vgg') or arch.startswith('alexnet'):
        input_size = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
    elif arch.startswith('resnet'):
        input_size = model.fc.in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = classifier
    elif arch.startswith('densenet'):
        input_size = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, output_size)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    return model


def validate_model(model, criterion, data_loader, device):
    """Validate the model on the validation set."""
    model.eval()
    loss = 0
    accuracy = 0
    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            loss += criterion(output, labels).item()
            ps = torch.exp(output)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return loss / len(data_loader), accuracy / len(data_loader)


def train_model(model, train_loader, valid_loader, criterion, optimizer, device, epochs=2):
    """Train the model."""
    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(epochs):
        for inputs, labels in train_loader:
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                validation_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():    
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)

                        validation_loss += batch_loss.item()

                        # Calculate accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}.. "
                    f"Train loss: {running_loss/print_every:.3f}.. "
                    f"Val loss: {validation_loss/len(valid_loader):.3f}.. "
                    f"Val accuracy: {accuracy/len(valid_loader):.3f}")
                running_loss = 0
                model.train()


def save_checkpoint(model, save_dir, arch, class_to_idx, hidden_units, optimizer, epochs):
    """Save the model checkpoint."""
    checkpoint = {
        'model_architecture': arch,
        'model_state_dict': model.state_dict(),
        'class_to_idx': class_to_idx,
        'hidden_units': hidden_units,
        'optimizer_state_dict': optimizer.state_dict(),
        'epochs': epochs
    }
    save_path = os.path.join(save_dir, 'model_checkpoint.pth')
    torch.save(checkpoint, save_path)
    print(f"Model saved to {save_path}")
    
    
def load_checkpoint(filepath):
    """Load a model checkpoint and rebuild the model."""

    checkpoint = torch.load(filepath)
    arch = checkpoint['model_architecture']
    hidden_units = checkpoint['hidden_units']
    
    if arch.startswith('vgg'):
        model = models.vgg16(pretrained=True)
    elif arch.startswith('densenet'):
        model = models.densenet121(pretrained=True)
    elif arch.startswith('resnet'):
        model = models.resnet18(pretrained=True)
    else:
        raise ValueError(f"Unsupported architecture: {arch}")
    
    # Build the model using the architecture specified in the checkpoint
    if arch.startswith('vgg') or arch.startswith('alexnet'):
        input_size = model.classifier[0].in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
    elif arch.startswith('resnet'):
        input_size = model.fc.in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.fc = classifier
    elif arch.startswith('densenet'):
        input_size = model.classifier.in_features
        classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_size, hidden_units)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc2', nn.Linear(hidden_units, 256)),
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(0.2)),
                          ('fc3', nn.Linear(256, 102)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
        model.classifier = classifier
    else:
        raise ValueError(f"Unsupported architecture: {arch}")

    # Load the state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Load class_to_idx mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    # Load optimizer state if needed (for further training)
    if 'optimizer' in checkpoint:
        optimizer = optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Set model to evaluation mode
    model.eval()
    
    return model


def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    image_tensor = helper.process_image(image_path).to(device)
    
    with torch.no_grad():
        logps = model(image_tensor)
        ps = torch.exp(logps)
    
    probs, indices = torch.topk(ps, topk)
    probs = probs.cpu().numpy().squeeze()
    indices = indices.cpu().numpy().squeeze()
    
    # Invert the class_to_idx dictionary to get a mapping from index to class
    idx_to_class = {v: k for k, v in model.class_to_idx.items()}
    classes = [idx_to_class[idx] for idx in indices]
    return probs, classes
