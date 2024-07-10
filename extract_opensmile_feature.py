# pip install senselab

from senselab.audio.tasks.features_extraction.opensmile import extract_opensmile_features_from_audios
from senselab.audio.data_structures.audio import Audio

#define the audio file path on local drive
audio_path = r"./bids_with_sensitive_recordings/sub-6ca9935e-13f3-4531-bc54-5237a9f8828e/ses-9C429A56-74A9-4F7E-AB9F-127E22AD18DA/audio/sub-6ca9935e-13f3-4531-bc54-5237a9f8828e_ses-9C429A56-74A9-4F7E-AB9F-127E22AD18DA_Audio-Check_rec-Audio-Check-1.wav"

audio = Audio.from_filepath(audio_path)
opensmile_features = extract_opensmile_features_from_audios([audio])

# print(len(opensmile_features[0]))
