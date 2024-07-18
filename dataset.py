
import os, random
import warnings
import librosa
#import parselmouth
#from parselmouth.praat import call
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import random
from sklearn.model_selection import train_test_split

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torchsummary import summary
from b2aiprep.dataset import VBAIDataset
from b2aiprep.process import Audio,specgram,melfilterbank
import torchaudio.transforms as T
#from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift, TimeMask

warnings.filterwarnings("ignore", category=UserWarning, message="PySoundFile failed. Trying audioread instead.")
warnings.filterwarnings("ignore", category=FutureWarning, message="librosa.core.audio.__audioread_load")

def get_person_session_pairs(vbai_dataset):

    qs = vbai_dataset.load_questionnaires('recordingschema')
    q_dfs = []
    for i, questionnaire in enumerate(qs):
        df = vbai_dataset.questionnaire_to_dataframe(questionnaire)
        df['dataframe_number'] = i
        q_dfs.append(df)
        i += 1
    recordingschema_df = pd.concat(q_dfs)
    recordingschema_df = pd.pivot(recordingschema_df, index='dataframe_number', columns='linkId', values='valueString')

    person_session_pairs = recordingschema_df[['record_id', 'recording_session_id']].to_numpy().astype(str)
    person_session_pairs = np.unique(person_session_pairs, axis=0).tolist()

    return person_session_pairs


def get_dataset(data_dir,target_diagnosis='voc_fold_paralysis',algo='DT',spec_gram=False,random_state=123):

    dataset = VBAIDataset(data_dir)

    participant_df = dataset.load_and_pivot_questionnaire('participant')
    all_identities = sorted(participant_df['record_id'].to_numpy().tolist())
    N = len(all_identities)
    
    '''
    train_identities = set(all_identities[:int(0.8*N)])
    val_identities = set(all_identities[int(0.8*N):int(0.9*N)])
    test_identities = set(all_identities[int(0.9*N):])
    '''

    train_identities, DT_test_identities = train_test_split(all_identities, test_size=0.15, random_state=random_state)
    val_identities, test_identities = train_test_split(DT_test_identities, test_size=0.5, random_state=random_state)

    print('train ids:', len(train_identities))
    print('val ids:', len(val_identities))
    print('test ids:', len(test_identities))

    #target_diagnosis = 'voc_fold_paralysis' #airway_stenosis

    person_session_pairs = get_person_session_pairs(dataset)

    print('Found {} person/session pairs'.format(len(person_session_pairs)))
    print('--------------------------')

    train_dataset = MyAudioDataset(train_identities, dataset,person_session_pairs,diagnosis_column=target_diagnosis,algo=algo,spec_gram=spec_gram,split='Train')
    val_dataset = MyAudioDataset(val_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo,spec_gram=spec_gram)
    test_dataset = MyAudioDataset(test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo,spec_gram=spec_gram)
    test_dataset_DT = MyAudioDataset(test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo='DT',spec_gram=False)
    DT_test_dataset = MyAudioDataset(DT_test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo='DT',spec_gram=False)
    #full_dataset = MyAudioDataset(all_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo,spec_gram=spec_gram)
    full_dataset = ''

    print("Train data size : " , len(train_dataset))
    print("Validation data size : ", len(val_dataset))
    print("Test data size : ", len(test_dataset))
    print("-----------------------")
    print("Test set for Decision Tree Algo : ", len(DT_test_dataset))
    #print("Lenght of full dataset : " , len(full_dataset))

    dataset_dict = {
    "VBAIDataset" : (dataset),
    "train_dataset": (train_dataset, train_identities),
    "val_dataset": (val_dataset, val_identities),
    "test_dataset": (test_dataset, test_identities, test_dataset_DT),
    "DT_test_dataset": (DT_test_dataset, DT_test_identities),
    "full_dataset": (full_dataset, all_identities)
    }

    return dataset_dict



class MyAudioDataset(torch.utils.data.Dataset):
    def __init__(self, identities, dataset, person_session_pairs, split='Test', diagnosis_column = 'voc_fold_paralysis', algo ='DT',spec_gram=False, segment_size=3):

        self.diagnosis_column = diagnosis_column
        self.algorithm = algo
        self.spec_gram = spec_gram
        self.split = split
        
        # Define gender mapping
        self.gender_mapping = {
            'Male gender identity': 0,
            'Female gender identity': 1,
            'no record': 2,
            'Non-binary or genderqueer gender identity': 3
        }
        self.reverse_gender_mapping = {v: k for k, v in self.gender_mapping.items()}

        self.site_mapping = {
            'Mt. Sinai': 0, 
            'VUMC': 1,
            'MIT': 2,
            'USF': 3,
            'WCM': 4
        }

        # Define bins for age ranges and labels
        self.bins = [0, 20, 40, 60, 80, 100]
        self.labels = [0, 1, 2, 3, 4]

        self.reverse_site_mapping = {v: k for k, v in self.site_mapping.items()}
        
        # get age and airway stenosis classification for all subjects
        participant_df = dataset.load_and_pivot_questionnaire('participant')
        demographics_df = dataset.load_and_pivot_questionnaire('qgenericdemographicsschema').drop_duplicates(subset='record_id', keep='first')
        
        age_data = participant_df[['record_id', 'age']].to_numpy()
        site_data = participant_df[['record_id', 'enrollment_institution']].to_numpy()
        
        gender_df = demographics_df[['record_id', 'gender_identity']]
        no_demographics_record = ['2af5afbc-82b1-4656-a203-a8d29b69d3ab', '5201d61d-4b67-482f-bddd-39466e63c4f4']
        new_row_df = pd.DataFrame({'record_id': no_demographics_record, 'gender_identity': ['no record', 'no record']})
        
        gender_data = pd.concat([gender_df, new_row_df], ignore_index=True).to_numpy()

        diagnosis_data = participant_df[['record_id', self.diagnosis_column]].to_numpy()
        
        age_dict = {}
        binned_age_dict = {}
        for person_id, age in age_data:
            binned_age = np.digitize(float(age), self.bins, right=True) - 1
            age_dict[str(person_id)] = float(age)
            binned_age_dict[str(person_id)] = self.labels[binned_age]

        gender_dict = {}
        for person_id, gender in gender_data:
            gender_dict[str(person_id)] = self.gender_mapping.get(str(gender), -1)  # Default to -1 if not found
        
        site_dict = {}
        for person_id, site in site_data:
            site_dict[str(person_id)] = self.site_mapping.get(str(site), -1)  # Default to -1 if not found

        diagnosis_dict = {}
        for person_id, dgnsis in diagnosis_data:
            diagnosis_dict[str(person_id)] = float(dgnsis)
        
        self.audio_files = []
        self.feature_files = []
        self.age = []
        self.binned_age = []
        self.gender = []
        self.site = []
        self.diagnosis = []
        self.aud_segments = []
        
        for person_id, session_id in person_session_pairs:
            if person_id not in identities:
                continue

            aud_feature_files = sorted([str(path) for path in dataset.find_audio_features(person_id, session_id) if "Audio-Check" not in str(path)])

            if self.spec_gram:
                # self.aud_augment = Compose([
                #     AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.015, p=0.5),
                #     TimeStretch(min_rate=0.8, max_rate=1.25, p=0.5),
                #     PitchShift(min_semitones=-4, max_semitones=4, p=0.5),
                #     Shift(p=0.5),
                #     #SpecFrequencyMask(p=0.5),
                #     TimeMask(min_band_part=0.1, max_band_part=0.4, p=0.5),
                #     ])
                self.n_fft = 1024
                self.sample_rate = 16000
                self.spec_transform = T.Spectrogram(n_fft=self.n_fft, power=2)
                self.mel_scale_transform = T.MelScale(n_mels=20, sample_rate=self.sample_rate, n_stft=self.n_fft // 2 + 1)

                self.spec_aug = None
                if self.split.upper()=='TRAIN':
                    self.spec_aug = torch.nn.Sequential(
                        T.TimeStretch(0.8,fixed_rate=True),
                        T.FrequencyMasking(freq_mask_param=80),
                        T.TimeMasking(time_mask_param=80),
                    )
                
                aud_files = sorted([str(path) for path in dataset.find_audio(person_id, session_id) if "Audio-Check" not in str(path)])
                #self.audio_files += aud_files
                self.segment_size = segment_size
                self.sample_rate = 16000
                self.segment_duration = self.segment_size * self.sample_rate
                self.overlap = 0.1
                self.step_size = int(self.segment_duration * (1 - self.overlap))
                #self.win_length = 30
                #self.hop_length = 10
                #self.nfft = 512
                self._generate_aud_segments(person_id,aud_files,aud_feature_files,age_dict,binned_age_dict,gender_dict,site_dict,diagnosis_dict)

            else:
                self.feature_files += aud_feature_files
                self.age += [age_dict[person_id]]*len(aud_feature_files)
                self.binned_age += [binned_age_dict[person_id]]*len(aud_feature_files)
                self.gender += [gender_dict[person_id]]*len(aud_feature_files)
                self.site += [site_dict[person_id]]*len(aud_feature_files)
                self.diagnosis += [diagnosis_dict[person_id]]*len(aud_feature_files)

        if self.spec_gram:
            assert len(self.aud_segments) == len(self.feature_files) == len(self.age) == len(self.gender) == len(self.site) == len(self.diagnosis)
        else:
            len(self.feature_files) == len(self.age) == len(self.gender) == len(self.site) == len(self.diagnosis)

    def _generate_aud_segments(self, person_id, aud_files, aud_feature_files, age_dict, binned_age_dict, gender_dict, site_dict, diagnosis_dict):

        
        # mel_transform = T.MelSpectrogram(
        #     sample_rate=16000,
        #     n_fft=int(20*16000/1000),  # Typically used value
        #     win_length=20,  # Typically used value, equal to n_fft to avoid extra padding
        #     hop_length=10,  # 10 ms hop length for a 16 kHz sample rate
        #     n_mels=20,  # Number of mel bins
        # )
        # mel_transform = T.MelSpectrogram(
        #     sample_rate=16000,
        #     #n_fft=512,  # Number of FFT components
        #     #win_length=512,  # Size of FFT window
        #     #hop_length=160,  # Number of samples between frames
        #     n_mels=20,  # Number of Mel bins
        #     )


        for aud_file, ftr_file in zip(aud_files, aud_feature_files):
            audio = Audio.from_file(aud_file)
            audio = audio.to_16khz()

            if audio.signal.size(0) < self.segment_duration:
                # Pad only if the segment size is larger than the audio signal
                audio.signal = torch.nn.functional.pad(audio.signal, (0, self.segment_duration - audio.signal.size(0)), mode='constant', value=0)

            # Generate mel-spectrograms for overlapping segments
            for start in range(0, len(audio.signal) - self.segment_duration + 1, self.step_size):
                segment = audio.signal[start:start + self.segment_duration] # 480000, channel
                #segment = segment.permute(1, 0) # channel, 480000
                # # apply augmentations to raw audio segment
                # # if self.split.upper()=='TRAIN':
                # #     segment = self.aud_augment(segment.numpy(),16000)
                # #     segment = torch.tensor(segment) # shape: (channel,16000*duration) =>  (1,48000)
                # mel_specgram_segment = mel_transform(segment) # torch.Size([1, 20, 4801] => channel,n_mels,sample_rate
                # mel_specgram_segment = mel_specgram_segment.squeeze(0) # torch.Size([20, 4801] => n_mels,sample_rate
                # # if random.random() < 0.5:
                # #     segment = T.FrequencyMasking(freq_mask_param=80)(mel_specgram_segment)

                self.aud_segments.append(segment.squeeze(1)) # 480000 => Removing channel dimension
                self.audio_files += [aud_file] #[aud_file]*1
                self.feature_files += [ftr_file]
                self.age += [age_dict[person_id]] #[age_dict[person_id]]*1
                self.binned_age += [binned_age_dict[person_id]] #[binned_age_dict[person_id]]*1
                self.gender += [gender_dict[person_id]] #[gender_dict[person_id]]*1
                self.site += [site_dict[person_id]] #[site_dict[person_id]]*1
                self.diagnosis += [diagnosis_dict[person_id]] #[diagnosis_dict[person_id]]*1
        
    def __len__(self):
        return len(self.feature_files)
    
    def map_gender_back(self, gender_numeric):
        return self.reverse_gender_mapping.get(gender_numeric, 'unknown')
        
    def map_site_back(self, site_numeric):
        return self.reverse_site_mapping.get(site_numeric, 'unknown')

    def set_algorithm_type(self, algo='DT'):
        self.algorithm = algo
        
    def __getitem__(self, idx):

        feature = torch.load(self.feature_files[idx])
        opensmile_feature = feature['opensmile']
        yamnet_embedding = feature['speaker_embedding']
        age = self.age[idx]
        gender = self.gender[idx]
        site = self.site[idx]
        binned_age = self.binned_age[idx]
        diagnosis = self.diagnosis[idx]
        
        if self.algorithm.upper() == 'DL':
            try:
                aud_seg = self.aud_segments[idx]
                #mel_spec_aug = self.aud_augment(samples=mel_spec.numpy(), sample_rate=16000)
                #mel_spec_aug = torch.tensor(mel_spec_aug)
                #return mel_spec, yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)
                # spec = self.spec_transform(aud_seg)
                # if self.spec_aug and random.random() < 0.5:
                #     spec = self.spec_aug(spec)
                # mel_spec = self.mel_scale_transform(spec)

                # return mel_spec, yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)
                return aud_seg, yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)

            except (AttributeError, IndexError):
                mel_spec = None

            return yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)

            # if self.spec_gram:
            #     mel_spec = self.mel_spectrograms[idx]
            #     return mel_spec, float(age), gender, site, float(binned_age) , float(diagnosis)

            # return yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)

        else:
            return opensmile_feature, age, gender, site, binned_age , diagnosis



### prepares data for visualization and non-dl algorithms

def create_open_smile_df(audio_dataset,include_GAS=True,diagnosis_column='voc_fold_paralysis',algo='DT'):
    
    opensmile_features = []
    ages = []
    genders = []
    sites = []
    GENDERS_org = []
    SITES_org = []
    AGE_binned = []
    diagnosis = []

    for i in range(len(audio_dataset)):
        opensmile_feature,age,gender,site,binned_age,diagnosis_label = audio_dataset[i]
        opensmile_features.append(opensmile_feature.squeeze())
        ages.append(age)
        genders.append(gender)
        sites.append(site)
        GENDERS_org.append(audio_dataset.map_gender_back(gender))
        SITES_org.append(audio_dataset.map_site_back(site))
        AGE_binned.append(binned_age)
        diagnosis.append(diagnosis_label)
    
    # Convert to DataFrame
    opensmile_df = pd.DataFrame(opensmile_features)
    opensmile_df['age'] = ages
    opensmile_df['gender'] = genders
    opensmile_df['site'] = sites
    opensmile_df['GENDER_org'] = GENDERS_org
    opensmile_df['SITE_org'] = SITES_org
    opensmile_df['AGE_bin'] = AGE_binned
    opensmile_df[diagnosis_column] = diagnosis
    
    
    # Standardize the opensmile features
    if include_GAS:
        feature_columns = opensmile_df.columns[:-4]
        label_columns = opensmile_df.columns[-4:]
    
    else:
        feature_columns = opensmile_df.columns[:-7]
        label_columns = opensmile_df.columns[-7:]
    
    return opensmile_df, feature_columns, label_columns