
# you can import pretrained models for experimentation & add your own created models
from torchvision.models import resnet18, resnet34, resnet50, resnet101, resnet152, vgg16, vgg19, inception_v3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.transforms as T
from b2aiprep.process import Audio,specgram

#------------------------------------------------
class MelSpecModel(torch.nn.Module):
    def __init__(
        self,
        #input_freq=16000,
        resample_freq=16000,
        n_fft=1024,
        n_mel=20,
        stretch_factor=0.2,
        num_classes=1
    ):
        super().__init__()
        self.num_classes = num_classes
        #self.resample = Resample(orig_freq=input_freq, new_freq=resample_freq)

        self.spec = T.Spectrogram(n_fft=n_fft, power=2)

        self.spec_aug = torch.nn.Sequential(
            T.TimeStretch(stretch_factor, fixed_rate=True),
            T.FrequencyMasking(freq_mask_param=20),
            T.TimeMasking(time_mask_param=20),
        )

        self.mel_scale = T.MelScale(n_mels=n_mel, sample_rate=resample_freq, n_stft=n_fft // 2 + 1)

        self.classifier = resnet18("IMAGENET1K_V1")
        #self.classifier = resnet18()
        self.classifier.fc = nn.Linear(512*1*1, self.num_classes)
    
    def normalize_spec(self,spec):
        return (spec - spec.mean()) / spec.std()

    def to_log_scale(self,spec):
        return torch.log(spec + 1e-6)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Resample the input
        #resampled = self.resample(waveform)

        # Convert to power spectrogram
        spec = self.spec(waveform)
        spec = self.to_log_scale(spec)
        # Apply SpecAugment
        if self.training:
            spec = self.spec_aug(spec)

        # Convert to mel-scale
        mel = self.mel_scale(spec)
        mel = self.normalize_spec(mel)

        bs, h, w = mel.shape
        mel = mel.unsqueeze(1).expand(bs,3,h,w)

        x = self.classifier(mel)

        return x


def get_models(args,num_classes = 1,spec_gram=True,pretrained=True):

    if spec_gram:

        if pretrained:
            #model = resnet18("IMAGENET1K_V1")
            model = MelSpecModel(num_classes=1)
        #model_18.conv1 = nn.Conv2d(128, 64, kernel_size=7, stride=2, padding=3, bias=False)
        #model.fc = nn.Linear(512*1*1, num_classes)
        
    else:
        model = FCModel_4(hs=128)

    return model

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




#------------------------------------------------

class FCModel_5(torch.nn.Module):
    def __init__(self, hs=64, dropout=0.5):
        
        super(FCModel_5, self).__init__()
        self.dropout = dropout

        self.lin1 = torch.nn.Linear(192, hs) # input size -> hidden size
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 32)
        self.lin4 = torch.nn.Linear(32, 1) # hidden size -> output size

        self.fcs = nn.Sequential(
            self.lin1,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin2,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            torch.nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin3,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin4,

            )
    
    def forward(self, x):

        x = self.fcs(x)

        return x


class FCModel_4(torch.nn.Module):
    def __init__(self, hs=128, dropout=0.5):
        
        super(FCModel_4, self).__init__()
        self.dropout = dropout

        self.lin1 = torch.nn.Linear(192, hs) # input size -> hidden size
        self.lin2 = torch.nn.Linear(128, 64)
        self.lin3 = torch.nn.Linear(64, 32)
        self.lin4 = torch.nn.Linear(32, 1) # hidden size -> output size

        self.fcs = nn.Sequential(
            self.lin1,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin2,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin3,
            nn.ReLU(),
            nn.Dropout(p=self.dropout),

            self.lin4,

            )
    
    def forward(self, x):

        x = self.fcs(x)

        return x
    
    