# import the necessary packages
from PIL import Image
import numpy as np
import cv2
import os
from time import sleep
from tqdm import tqdm
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy
from torch.autograd import Variable
import torch.nn.functional as F

class TrainImageDataset(Dataset):
    def __init__(self, img_folder, labels, map, transform=None):
        self.transform = transform
        self.img_folder = img_folder
        self.labels_frame = pd.read_csv(labels)
        self.names_frame = pd.read_csv(map)
        self.labels = self.labels_frame.iloc[:,2] #names
        self.names_list = self.names_frame.iloc[:,1] #names in order
        self.all_imgs = os.listdir(self.img_folder)
        self.true_labels = []
        self.names = []
        for i in self.all_imgs:
            #print(self.labels)
            name = self.labels[int(i.split('.')[0])]
            id = list(self.names_list).index(name) # may take long
            self.true_labels.append(id)
            self.names.append(name)
    def __len__(self):
        return len(self.all_imgs)
    def __getitem__(self,idx):
        path = self.img_folder + "/" + str(self.all_imgs[idx])
        image_raw = Image.open(path).convert('RGB')
        if self.transform:
            image_raw = self.transform(image_raw)
        #return {'image':image_raw, 'label':self.true_nums[idx], 'name':self.true_labels[idx], 'filename':self.all_imgs[idx]}
        return image_raw, self.true_labels[idx], self.names[idx], str(self.all_imgs[idx])
        #return (image_raw, self.true_nums[idx])
    

if __name__ == '__main__':
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    else:
        print("Not Using Cuda")

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_dataset = TrainImageDataset("train_proc", "train.csv", "category.csv",train_transform)
    validation_dataset = TrainImageDataset("val_proc", "train.csv", "category.csv",val_transform)

    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle=True, num_workers=4)
    val_loader = DataLoader(validation_dataset, batch_size = 8, shuffle=True, num_workers=4)

    best_acc = 0

    print(len(train_dataset))
    print(len(validation_dataset))

    vgg16 = models.vgg16(weights='DEFAULT')
    #vgg16 = models.vgg16()

    # Freeze training for all layers
    for param in vgg16.features.parameters():
        param.require_grad = False

    # Newly created modules have require_grad=True by default
    num_features = vgg16.classifier[6].in_features
    features = list(vgg16.classifier.children())[:-1] # Remove last layer


    #features.extend([nn.Sequential(
    #nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.4),
    #nn.Linear(256, 100), nn.LogSoftmax(dim=1))]) # Add our layer with 4 outputs

    features.extend([nn.Linear(num_features, 100)]) # Add our linear layer with 4 outputs

    vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    #vgg16.load_state_dict(torch.load('fun_new_try.pth')) // for further training

    print(vgg16)

    if use_gpu:
        vgg16.cuda() #.cuda() will move everything to the GPU side

    
    
    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(vgg16.parameters(), lr=0.001, momentum=0.9)
    #exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

    def train(classifier, epoch):
        classifier.train() # we need to set the mode for our model

        for batch_idx, (images, targets, _, _) in enumerate(train_loader):

            images = images.cuda()
            targets = targets.cuda()
            optimizer.zero_grad()
            output = classifier(images)
            #print(output)
            loss = F.cross_entropy(output, targets) # Here is a typical loss function (negative log likelihood)
            loss.backward()
            optimizer.step()

            if batch_idx % 100 == 0: # We visulize our output every 10 batches
                print(f'Epoch {epoch}: [{batch_idx*len(images)}/{len(train_loader.dataset)}] Loss: {loss.item()}')

    def test(classifier, epoch):

        classifier.eval() # we need to set the mode for our model

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for images, targets, _, _ in val_loader:
                images = images.cuda()
                targets = targets.cuda()
                output = classifier(images)
                test_loss += F.cross_entropy(output, targets, reduction='sum').item()
                pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
                correct += pred.eq(targets.data.view_as(pred)).sum() # sum up the corrected samples

        test_loss /= len(val_loader.dataset)

        print(f'Test result on epoch {epoch}: Avg loss is {test_loss}, Accuracy: {100.*correct/len(val_loader.dataset)}%')

        return correct/len(val_loader.dataset)
    

    max_epoch = 20
    for epoch in range(1, max_epoch+1):
        train(vgg16, epoch)
        new_acc = test(vgg16, epoch)
        if (new_acc > best_acc):
            print("New Best Accuracy")
            best_acc = new_acc
            best_model_wts = copy.deepcopy(vgg16.state_dict())
    
    vgg16.load_state_dict(best_model_wts)
    print(test(vgg16,1))


    torch.save(vgg16.state_dict(), 'big_classifier_new.pth')