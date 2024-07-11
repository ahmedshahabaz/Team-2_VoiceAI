# pip install senselab

from senselab.audio.tasks.features_extraction.opensmile import extract_opensmile_features_from_audios
from senselab.audio.data_structures.audio import Audio

#define the audio file path on local drive
audio_path = "Audio.wav"

audio = Audio.from_filepath(audio_path)
opensmile_features = extract_opensmile_features_from_audios([audio])

# print(len(opensmile_features[0]))
