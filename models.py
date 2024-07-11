
# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F



#------------------------------------------------


def get_models(args, num_classes = 1):

    # model_18 = resnet18(pretrained = False)
    # model_18.fc = nn.Linear(512*1*1, num_classes)

    # model_16 = vgg16(pretrained = False)
    # model_16.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    # model_16_1 = vgg16(pretrained = False)
    # model_16_1.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    # model_34 = resnet34(pretrained = False)
    # model_34.fc = nn.Linear(512*1*1,num_classes)

    # model_50 = resnet50(pretrained = False)
    # model_50.fc = nn.Linear(2048,num_classes)

    # model_50_1 = resnet50(pretrained = False)
    # model_50_1.fc = nn.Linear(2048,num_classes)

    # model_101 = resnet101(pretrained = True)
    # model_101.fc = nn.Linear(2048,num_classes)

    # '''
    # model_101_1 = resnet101(pretrained = True)
    # model_101_1.fc = nn.Linear(2048,num_classes)

    # model_101_2 = resnet101(pretrained = True)
    # model_101_2.fc = nn.Linear(2048,num_classes)
    # '''

    # models = [model_18, model_16, model_16_1, model_34, model_50, model_50_1, model_101]#, model_101_1, model_101_2]

    model = FCModel(hs=64)
    return model


#------------------------------------------------

class FCModel(torch.nn.Module):
    def __init__(self, hs=64, dropout=0.5):
        
        super(FCModel, self).__init__()
        self.dropout = dropout

        self.lin1 = torch.nn.Linear(192, hs) # input size -> hidden size
        self.lin2 = torch.nn.Linear(hs, 32)
        self.lin3 = torch.nn.Linear(32, 1) # hidden size -> output size

        self.fcs = nn.Sequential(
            self.lin1,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin2,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin3,

            )
    
    def forward(self, x):

        x = self.fcs(x)

        return x
    
    