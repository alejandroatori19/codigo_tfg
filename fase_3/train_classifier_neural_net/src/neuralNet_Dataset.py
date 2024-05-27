from torch.utils.data import Dataset
import torch
import torch.nn as nn
import cv2 as cv


class SimilarityDataset(Dataset):
     def __init__(self, frame1_paths, frame2_paths, labels, input_shape, transform=None):
          self.frame1_paths = frame1_paths
          self.frame2_paths = frame2_paths
          self.labels = labels
          self.input_shape = input_shape
          self.transform = transform

     def __len__(self):
          return len(self.frame1_paths)

     def __getitem__(self, idx):
          frame1 = cv.imread(self.frame1_paths[idx])
          frame2 = cv.imread(self.frame2_paths[idx])
          label = self.labels[idx]

          frame1 = cv.resize(frame1, (self.input_shape[1], self.input_shape[0]))
          frame2 = cv.resize(frame2, (self.input_shape[1], self.input_shape[0]))

          if self.transform:
               frame1 = self.transform(frame1)
               frame2 = self.transform(frame2)

          return frame1, frame2, label
     

     def get_data_without_transform (self, idx):
          frame1 = cv.resize (cv.imread(self.frame1_paths[idx]), (self.input_shape[1], self.input_shape[0]))
          frame2 = cv.resize (cv.imread(self.frame2_paths[idx]), (self.input_shape[1], self.input_shape[0]))
          label = self.labels[idx]
          
          return frame1, frame2, label

class SimilarityModel(nn.Module):
     def __init__(self, input_shape):
          super(SimilarityModel, self).__init__()

          self.shared_layers = nn.Sequential(
               nn.Conv2d(input_shape[2], 32, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2, stride=2),
               nn.Conv2d(32, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.MaxPool2d(kernel_size=2, stride=2),
               nn.Conv2d(64, 64, kernel_size=3, padding=1),
               nn.ReLU(),
               nn.Flatten(),
               nn.Linear(64 * (input_shape[0] // 4) * (input_shape[1] // 4), 64),
               nn.ReLU()
          )

          self.fc = nn.Linear(64 * 2, 1)
          self.sigmoid = nn.Sigmoid()

     def forward(self, x1, x2):
          x1 = self.shared_layers(x1)
          x2 = self.shared_layers(x2)
          x = torch.cat((x1, x2), dim=1)
          x = self.fc(x)
          x = self.sigmoid(x)
          
          return x