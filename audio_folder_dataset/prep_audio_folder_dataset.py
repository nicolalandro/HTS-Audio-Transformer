import os
import json
import librosa
import numpy as np
from tqdm import tqdm

train_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound/train'
test_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound/test'
out_class_json = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound/classes.json'

classes = os.listdir(train_path)
classes.sort()

with open(out_class_json, 'w') as f:
    json.dump({i:v for i, v in enumerate(classes)}, f)

class2index = {v:i for i, v in enumerate(classes)}

output_dict = [[] for _ in range(10)]
max_audio_len = 10000000
max = 0
for fold_number, fold_path in enumerate([train_path, test_path]):
    for c in tqdm(os.listdir(fold_path)):
        class_id = int(class2index[c])
        for file_name in os.listdir(os.path.join(fold_path, c)):
            file_path = os.path.join(fold_path, c, file_name)
            
            ## loading audio file
            audio, sr = librosa.load(file_path, sr=32000, res_type='kaiser_fast')

            if len(audio) > max:
                max = len(audio)
                print('rel max:', max)
            audio = np.pad(audio, (0, max_audio_len - len(audio)), 'constant', constant_values=0)

            output_dict[fold_number].append(
            {
                "name": file_name,
                "target": int(class_id),
                "waveform": audio
            }
            )
print('real max:', max, 'pad:', max_audio_len)