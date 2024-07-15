
import os
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
from b2aiprep.process import Audio, specgram

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


def get_dataset(data_dir,target_diagnosis='voc_fold_paralysis',algo='DT',random_state=123):

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

    train_dataset = MyAudioDataset(train_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo)
    val_dataset = MyAudioDataset(val_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo)
    test_dataset = MyAudioDataset(test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo)
    test_dataset_DT = MyAudioDataset(test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo='DT')
    DT_test_dataset = MyAudioDataset(DT_test_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo='DT')
    full_dataset = MyAudioDataset(all_identities, dataset, person_session_pairs,diagnosis_column=target_diagnosis,algo=algo)

    print("Train data size : " , len(train_dataset))
    print("Validation data size : ", len(val_dataset))
    print("Test data size : ", len(test_dataset))
    print("-----------------------")
    print("Test set for Decision Tree Algo : ", len(DT_test_dataset))
    print("Lenght of full dataset : " , len(full_dataset))

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
    def __init__(self, identities, dataset, person_session_pairs, diagnosis_column = 'voc_fold_paralysis', algo ='DT' , segment_size=3):
        
        self.segment_size = segment_size
        self.diagnosis_column = diagnosis_column
        self.algorithm = algo
        
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
        
        self.feature_files = []
        self.age = []
        self.binned_age = []
        self.gender = []
        self.site = []
        self.diagnosis = []
        
        for person_id, session_id in person_session_pairs:
            if person_id not in identities:
                continue
            
            audio_features = [str(path) for path in dataset.find_audio_features(person_id, session_id) if "Audio-Check" not in str(path)]
            self.feature_files += audio_features
            self.age += [age_dict[person_id]]*len(audio_features)
            self.binned_age += [binned_age_dict[person_id]]*len(audio_features)
            self.gender += [gender_dict[person_id]]*len(audio_features)
            self.site += [site_dict[person_id]]*len(audio_features)
            self.diagnosis += [diagnosis_dict[person_id]]*len(audio_features)
        
        assert len(self.feature_files) == len(self.age) == len(self.gender) == len(self.site) == len(self.diagnosis)
        
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
            return yamnet_embedding, float(age), gender, site, float(binned_age) , float(diagnosis)

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