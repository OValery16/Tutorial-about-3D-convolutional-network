import torch
import torch.nn as nn
import math
from functools import partial
from torch.autograd import Variable
    
class ConvColumn5(nn.Module):

    def __init__(self, num_classes):

        
        super(ConvColumn5, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5=nn.Conv3d(256, 256, kernel_size=(1, 3, 3), padding=0)
        
        self.fc5 = nn.Linear(256, 256)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(256)
        self.drop=nn.Dropout(p=0.15)        
        self.fc6 = nn.Linear(256, 124)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(124)
        
        self.drop=nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(124, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x,x1
    
class ConvColumn6(nn.Module):

    def __init__(self, num_classes):

        
        super(ConvColumn6, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5=nn.Conv3d(256, 512, kernel_size=(1, 3, 3), padding=0)
        
        self.fc5 = nn.Linear(512, 512)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(512)
        self.drop=nn.Dropout(p=0.15)        
        self.fc6 = nn.Linear(512, 256)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(256)
        
        self.drop=nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(256, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 3, 3), padding=0),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 3, 3), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x,x1
    
class ConvColumn7(nn.Module):

    def __init__(self, num_classes):

        
        super(ConvColumn7, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5 = self._make_conv_layer(256, 512)
        self.conv_layer6 = self._make_conv_layer2(512, 512)
        
        #self.conv_layer7=nn.Conv3d(256, 512, kernel_size=(1, 2, 2), padding=0)
        
        self.fc5 = nn.Linear(2048, 4096)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(4096)
        self.drop=nn.Dropout(p=0.15)        
        self.fc6 = nn.Linear(4096, 4096)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(4096)
        
        self.drop=nn.Dropout(p=0.15)
        self.fc7 = nn.Linear(4096, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def _make_conv_layer2(self, in_c, out_c):
        conv_layer = nn.Sequential(
           nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm3d(out_c),
           nn.LeakyReLU(),
           nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x=self.conv_layer6(x)
        #print(x.size())

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x,x1
    
class ConvColumn8(nn.Module):

    def __init__(self, num_classes):

        
        super(ConvColumn8, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5 = self._make_conv_layer(256, 512)
        self.conv_layer6 = self._make_conv_layer2(512, 512)
        
        #self.conv_layer7=nn.Conv3d(256, 512, kernel_size=(1, 2, 2), padding=0)
        
        self.fc5 = nn.Linear(2048, 4096)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(4096)
        self.drop=nn.Dropout(p=0.6)        
        self.fc6 = nn.Linear(4096, 4096)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(4096)
        
        self.drop=nn.Dropout(p=0.6)
        self.fc7 = nn.Linear(4096, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def _make_conv_layer2(self, in_c, out_c):
        conv_layer = nn.Sequential(
           nn.Conv3d(in_c, out_c, kernel_size=3, stride=1, padding=1),
           nn.BatchNorm3d(out_c),
           nn.LeakyReLU(),
           nn.MaxPool3d((2, 2, 2))
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x=self.conv_layer6(x)
        #print(x.size())

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x,x1

class ConvColumn9(nn.Module):

    def __init__(self, num_classes):

        
        super(ConvColumn9, self).__init__()

        self.conv_layer1 = self._make_conv_layer(3, 32)
        self.conv_layer2 = self._make_conv_layer(32, 64)
        self.conv_layer3 = self._make_conv_layer(64, 124)
        self.conv_layer4 = self._make_conv_layer(124, 256)
        self.conv_layer5 = self._make_conv_layer(256, 512)
        self.conv_layer6 = self._make_conv_layer2(512, 1024)
        
        #self.conv_layer7=nn.Conv3d(256, 512, kernel_size=(1, 2, 2), padding=0)
        
        self.fc5 = nn.Linear(1024, 4096)
        self.relu = nn.LeakyReLU()
        self.batch0=nn.BatchNorm1d(4096)
        self.drop=nn.Dropout(p=0.6)        
        self.fc6 = nn.Linear(4096, 4096)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(4096)
        
        self.drop=nn.Dropout(p=0.6)
        self.fc6 = nn.Linear(4096, 1024)
        self.relu = nn.LeakyReLU()
        self.batch1=nn.BatchNorm1d(1024)
        
        self.drop=nn.Dropout(p=0.6)
        self.fc7 = nn.Linear(1024, num_classes)

    def _make_conv_layer(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(2, 2, 2), padding=1),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        nn.MaxPool3d((2, 2, 2)),
        )
        return conv_layer
    
    def _make_conv_layer2(self, in_c, out_c):
        conv_layer = nn.Sequential(
        nn.Conv3d(in_c, out_c, kernel_size=(2, 2, 2), padding=0),
        nn.BatchNorm3d(out_c),    
        nn.LeakyReLU(),
        nn.Conv3d(out_c, out_c, kernel_size=(1, 3, 3), padding=0),
        nn.BatchNorm3d(out_c),
        nn.LeakyReLU(),
        )
        return conv_layer

    def forward(self, x):
        #print(x.size())
        x = self.conv_layer1(x)
        #print(x.size())
        x = self.conv_layer2(x)
        #print(x.size())
        x = self.conv_layer3(x)
        #print(x.size())
        x = self.conv_layer4(x)
        #print(x.size())
        x=self.conv_layer5(x)
        #print(x.size())
        x=self.conv_layer6(x)
        #print(x.size())

        x = x.view(x.size(0), -1)
        x = self.fc5(x)
        x = self.relu(x)
        x = self.batch0(x)
        x = self.drop(x)
        x = self.fc6(x)
        x = self.relu(x)
        x = self.batch1(x)
        x = self.drop(x)
        x1=x
        x = self.fc7(x)

        return x,x1


class Classifier(nn.Module):

    def __init__(self, num_classes):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(5*num_classes, num_classes)
        '''
        self.fc1 = nn.Linear(372, 512)
        self.relu = nn.LeakyReLU()
        self.drop=nn.Dropout(p=0.25)
        self.fc3 = nn.Linear(512, num_classes)
        
        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024, num_classes)
        '''
    def forward(self, x):
        #x=self.dropout(x)
        x = self.fc1(x)
        #x = self.drop(x)
        #x = self.fc3(x)
        '''
        
        x=self.relu(x)
        x=self.fc2(x)
        '''
        return x

if __name__ == "__main__":

    input_tensor = torch.autograd.Variable(torch.rand(32,3,84,84))
    model = ResNet(27) #ConvColumn(27).cuda()
    output = model(input_tensor) #model(input_tensor.cuda())
    print(output.size())