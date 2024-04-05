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
import csv

class TestImageDataset(Dataset):
    def __init__(self, img_folder, transform=None):
        self.transform = transform
        self.img_folder = img_folder
        self.paths = sorted(os.listdir(self.img_folder))
        self.newpaths = sorted({int(i.split('.')[0]) for i in self.paths})
        self.finalpaths = [] # goofy procedure to get the files in the right order
        for i in range(len(self.newpaths)):
            self.finalpaths.append(str(self.newpaths[i]) + ".jpg")
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.img_folder + "/" + str(self.finalpaths[idx])
        image_raw = Image.open(path).convert('RGB')
        if self.transform:
            image_raw = self.transform(image_raw)
        return image_raw, self.finalpaths[idx]

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
    map = pd.read_csv("category.csv")
    name_list = map.iloc[:,1] #names in order
    print(name_list[0])
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("Using CUDA")
    else:
        print("Not Using Cuda")
    
    validation_dataset = TrainImageDataset("val_proc", "train.csv", "category.csv",val_transform)
    test_dataset = TestImageDataset("test_proc", test_transform)
    val_loader = DataLoader(validation_dataset, batch_size = 8, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size = 8, shuffle=False, num_workers=4)
    
    #load in model
    vgg16 = models.vgg16()
    num_features = vgg16.classifier[6].in_features
    vgg16.classifier[6] = nn.Linear(num_features, 100)
    #vgg16.classifier[6] = nn.Sequential(nn.Linear(num_features, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 100), nn.LogSoftmax(dim=1))

    # Freeze training for all layers
    #for param in vgg16.features.parameters():
        #param.require_grad = False

    # Newly created modules have require_grad=True by default
    #num_features = vgg16.classifier[6].in_features
    #features = list(vgg16.classifier.children())[:-1] # Remove last layer
    #features.extend([nn.Linear(num_features, 100)]) # Add our layer with 4 outputs
    #vgg16.classifier = nn.Sequential(*features) # Replace the model classifier

    vgg16.load_state_dict(torch.load('big_classifier_new.pth'))
    print(vgg16)

    if use_gpu:
        vgg16.cuda() #.cuda() will move everything to the GPU side


    def test(classifier):

        classifier.eval() # we need to set the mode for our model
        fieldnames = ['Id', 'Category']
        with open("test_sub_new.csv", 'w') as csvfile:
            # creating a csv writer object
            csvwriter = csv.writer(csvfile,lineterminator='\n')
            csvwriter.writerow(fieldnames)
            with torch.no_grad():
                for i, (images, filenames) in enumerate(test_loader):
                    images = images.cuda()
                    #filenames = filenames.cuda()
                    output = classifier(images)
                    pred = output.data.max(1, keepdim=True)[1] # we get the estimate of our result by look at the largest class value
                    pred_size = torch.numel(pred)
                    for j in range(pred_size):
                        print(filenames[j])
                        data_row = [str(8*i + j), str(name_list[pred[j].item()])]
                        csvwriter.writerow(data_row)

        return

    def test_val(classifier, epoch):

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
    

    #test_val(vgg16, 1234)
    test(vgg16)
    #print(test_val(vgg16, 1234))