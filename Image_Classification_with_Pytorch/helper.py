import os
import argparse
import torch
import numpy as np
from torch import nn, optim
from torchvision import datasets, transforms, models
import torch.nn.functional as F
from torch.utils.data import DataLoader
from PIL import Image

def get_train_input_args():
    """Get input arguments for training from command line."""
    parser = argparse.ArgumentParser(description="Train a neural network on a dataset.")
    parser.add_argument('data_dir', type=str, help='Directory containing the dataset')
    parser.add_argument('--save_dir', type=str, default='.',
                        help='Directory to save the trained model')
    parser.add_argument('--arch', type=str, default='densenet121',
                        help='Model architecture to use (e.g., resnet18, vgg16)')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of hidden units in the classifier')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    
    return parser.parse_args()


def get_predict_input_args():
    """Get input arguments for prediction from command line."""
    parser = argparse.ArgumentParser(description="Predict flower name from an image along with the probability of that name")
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')
    
    return parser.parse_args()


def load_data(data_dir):
    """Load the dataset and apply transformations."""
    train_dir = os.path.join(data_dir, 'train')
    valid_dir = os.path.join(data_dir, 'valid')
    
    # Define transforms for the training and validation
    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load the datasets with ImageFolder
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    
    # Define the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=64)
    
    return train_loader, valid_loader, train_dataset.class_to_idx


def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    # Load the image using PIL
    pil_image = Image.open(image_path)

    # Resize the image while maintaining the aspect ratio
    pil_image.thumbnail((256, 256))

    # Crop the center 224x224 portion of the image
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = (pil_image.width + 224) / 2
    bottom = (pil_image.height + 224) / 2
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert the image to a NumPy array and normalize the pixel values to the range 0-1
    np_image = np.array(pil_image).astype(np.float32) / 255.0

    # Normalize the image using the specified means and standard deviations
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - means) / stds

    # Reorder the dimensions so that the color channel is the first dimension
    np_image = np_image.transpose((2, 0, 1))

    # Convert the NumPy array to a PyTorch tensor
    tensor_image = torch.tensor(np_image, dtype=torch.float32).unsqueeze(0)

    return tensor_image
