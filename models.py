
# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F



#------------------------------------------------


def get_models(args, num_classes = 19):

    model_18 = resnet18(pretrained = False)
    model_18.fc = nn.Linear(512*1*1, num_classes)

    model_16 = vgg16(pretrained = False)
    model_16.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    model_16_1 = vgg16(pretrained = False)
    model_16_1.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    model_34 = resnet34(pretrained = False)
    model_34.fc = nn.Linear(512*1*1,num_classes)

    model_50 = resnet50(pretrained = False)
    model_50.fc = nn.Linear(2048,num_classes)

    model_50_1 = resnet50(pretrained = False)
    model_50_1.fc = nn.Linear(2048,num_classes)

    model_101 = resnet101(pretrained = True)
    model_101.fc = nn.Linear(2048,num_classes)

    '''
    model_101_1 = resnet101(pretrained = True)
    model_101_1.fc = nn.Linear(2048,num_classes)

    model_101_2 = resnet101(pretrained = True)
    model_101_2.fc = nn.Linear(2048,num_classes)
    '''

    models = [model_18, model_16, model_16_1, model_34, model_50, model_50_1, model_101]#, model_101_1, model_101_2]

    return models


#------------------------------------------------

'''
Source: https://pytorch.org/hub/pytorch_vision_resnet/
http://www.youtube.com/watch?v=ACmuBbuXn20
https://www.youtube.com/watch?v=DkNIBBBvcPs
'''

class my_VGG(nn.Module):
    

    def __init__(self, input_channels = 3, num_classes=11, dropout=0.5):
        """
            A linear model for image classification.
        """

        super(my_VGG, self).__init__()
        self.architecture = [64,64,'M', 128,128, 'M', 256, 256, 256, 'M', 512, 512, 512]#, 'M', 512, 512, 512, 'M']
        self.input_channels = input_channels
        self.num_classes = num_classes
        self.dropout = dropout
    
        self.conv_layers = self.feature_extractor(self.architecture)
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fcs = nn.Sequential(

            nn.Linear(512 * 1 * 1, 4096, bias = True),
            nn.ReLU(inplace = False),
            nn.Dropout(p=self.dropout),

            nn.Linear(4096, 4096, bias= True),
            nn.ReLU(inplace = False),
            nn.Dropout(p=self.dropout),
            
            nn.Linear(4096, self.num_classes, bias = True)

            )

        # initialize parameters (write code to initialize parameters here)

    def forward(self, image):
        """
            feed-forward (vectorized) image into a linear model for classification.   
        """
        features = self.conv_layers(image)
        features = self.avgpool(features)
        features = features.reshape(features.shape[0], -1)
        features = self.fcs(features)

        return features

    def feature_extractor(self, architecture):

        layers = []
        in_chnl = self.input_channels

        for layer in architecture:

            if layer != 'M':
                out_chnl = layer

                layers += [ nn.Conv2d(in_channels = in_chnl, out_channels = out_chnl, kernel_size = (3,3), padding = 1),
                    nn.BatchNorm2d(layer), nn.ReLU(inplace = False)]
                in_chnl = layer

            else:
                layers += [nn.MaxPool2d(kernel_size = (2,2), stride = (2,2), padding = 0)]

        return nn.Sequential(*layers)

# =======================================

class Bottleneck(nn.Module):

    def __init__(self, in_channels, interim_channels, downsample = None, stride = 1):

        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, interim_channels, kernel_size = 1, 
            stride = 1, padding = 0, bias = False)
        self.bn1 = nn.BatchNorm2d(interim_channels)
        self.conv2 = nn.Conv2d(interim_channels, interim_channels, kernel_size = 3,
            stride = stride, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(interim_channels)
        self.conv3 = nn.Conv2d(interim_channels, interim_channels * 4, kernel_size = 1, 
            stride = 1, padding = 0, bias = False)
        self.bn3 = nn.BatchNorm2d(interim_channels * 4)
        self.relu = nn.ReLU(inplace = False)
        self.downsample = downsample

    def forward(self,x):

        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)

        if self.downsample is not None:
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x

class ResNet(nn.Module):

    def __init__(self, Bottleneck, blocks,  input_channels = 3,  num_classes=1000, dropout=0.5):
        """
            A linear model for image classification.
        """

        super(ResNet, self).__init__()

        self.in_channels = 64
        self.conv1 = nn.Conv2d(input_channels, 64, stride = 2, kernel_size = 7,
         padding = 3, bias = False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = False)
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

        #Residual Blocks
        self.layer1 = self.make_residual_blocks(Bottleneck, blocks[0], interim_channels = 64, stride = 1)
        self.layer2 = self.make_residual_blocks(Bottleneck, blocks[1], interim_channels = 128, stride = 2)
        #self.layer3 = self.make_residual_blocks(Bottleneck, blocks[2], interim_channels = 256, stride = 2)
        #self.layer4 = self.make_residual_blocks(Bottleneck, blocks[3], interim_channels = 512, stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        #self.fc = nn.Linear(512 * 1 * 1, num_classes, bias = True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.max_pool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        #x = self.layer3(x)
        #x = self.layer4(x)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def make_residual_blocks(self, Bottleneck, num_blocks, interim_channels, stride):

        downsample = None
        layers = []

        if stride !=1 or self.in_channels != interim_channels * 4:

            downsample = nn.Sequential(nn.Conv2d(self.in_channels, interim_channels * 4, 
                kernel_size = 1, stride = stride, bias = False),
            nn.BatchNorm2d(interim_channels * 4))

        layers.append(Bottleneck(self.in_channels, interim_channels, downsample, stride))
        
        self.in_channels = interim_channels * 4

        for i in range(num_blocks-1):
            layers.append(Bottleneck(self.in_channels, interim_channels))

        return nn.Sequential(*layers)



class my_ResNet(nn.Module):

    def __init__(self,architecture = 50, num_classes = 19, dropout = .5):

        super(my_ResNet, self).__init__()

        if(architecture == 50):
            self.resnet_model = ResNet(Bottleneck, blocks = [3, 4, 6, 3])

        elif(architecture == 101):
            self.resnet_model = ResNet(Bottleneck, blocks = [3, 4, 23, 3])

        if(architecture == 152):
            self.resnet_model = ResNet(Bottleneck, blocks = [3, 8, 36, 3])


        
        self.resnet_model.fc = nn.Sequential(

            #nn.Linear(512 * 1 * 1, 4096, bias = True),
            #nn.ReLU(inplace = False),
            #nn.Dropout(p=dropout),
            
            #nn.Linear(4096, 4096, bias= True),
            #nn.ReLU(inplace = False),
            #nn.Dropout(p=dropout),
            
            nn.Linear(512, num_classes, bias = True)

            )
        

    def forward(self, x):

        x = self.resnet_model(x)
        return x
