from distutils.command.config import config
import os
import json
import random
import librosa
import numpy as np
from tqdm import tqdm

train_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech/train'
test_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech/test'
out_class_json = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech/classes.json'
out_prepared_dataset = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech/esc50_urbansound_audioset_softech.npy'
dataset_path = "/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech"
# for scv2 format
train_set = "scv2_train.npy"
test_set =  "scv2_test.npy"

SR = 16000
MAX_SEC = 10
max_audio_len = MAX_SEC * SR

classes = os.listdir(train_path)
classes.sort()

with open(out_class_json, 'w') as f:
    json.dump({i:v for i, v in enumerate(classes)}, f)

class2index = {v:i for i, v in enumerate(classes)}

# output_dict = [[] for _ in range(10)] # init for 10 folds


# max = 0
for fold_number, fold_path in enumerate([(train_path, train_set), (test_path, test_set)]):
    output_dict = [] # init for train and test in scv2 format
    # use the number of files in the main class to cut the other classes
    # Change this value if you want
    max_files_per_class = len(os.listdir(os.path.join(fold_path[0], 'airplane')))
    # max_files_per_class = 10
    for c in tqdm(os.listdir(fold_path[0])):
        class_id = int(class2index[c])
        listfiles = os.listdir(os.path.join(fold_path[0], c))
        if len(listfiles) > max_files_per_class:
            random.shuffle(listfiles)
            listfiles = listfiles[:max_files_per_class]
        for file_name in tqdm(listfiles):
            file_path = os.path.join(fold_path[0], c, file_name)
            
            ## load audio file
            audio, sr = librosa.load(file_path, sr=SR, res_type='kaiser_fast')

            # if len(audio) > max:
            #     max = len(audio)
            #     print('rel max:', max)
            if len(audio) > max_audio_len:
                audio = audio[:max_audio_len]
            else:
                audio = np.pad(audio, (0, max_audio_len - len(audio)), 'constant', constant_values=0)

            # output_dict[fold_number].append(
            output_dict.append(
            {
                "name": file_name,
                "target": int(class_id),
                "waveform": audio
            }
            )
    file_path = os.path.join(dataset_path, fold_path[1])
    np.save(file_path, output_dict)
    print("Created pickle file in :", file_path)
# print('real max:', max, 'pad:', max_audio_len)


# np.save(out_prepared_dataset, output_dict)
# print("Created pickle file in :", out_prepared_dataset)
