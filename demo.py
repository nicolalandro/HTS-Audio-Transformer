import os
import config

import torch
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
import librosa
import numpy as np
import json

SR = 24000
MAX_SEC = 10
max_audio_len = MAX_SEC * SR

class_mapping_json = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset_softech/classes.json'

with open(class_mapping_json, 'r') as f:
  class_mapping = json.load(f) # {"0": "Air brake", "1": "Air horn, truck horn",


sed_model = HTSAT_Swin_Transformer(
    spec_size=config.htsat_spec_size,
    patch_size=config.htsat_patch_size,
    in_chans=1,
    num_classes=config.classes_num,
    window_size=config.htsat_window_size,
    config=config,
    depths=config.htsat_depth,
    embed_dim=config.htsat_dim,
    patch_stride=config.htsat_stride,
    num_heads=config.htsat_num_head
)

model = SEDWrapper(
    sed_model=sed_model,
    config=config,
    dataset=None
)

# ckpt = torch.load(config.resume_checkpoint, map_location="cpu")
model_path = 'saved_training/us+esc50_32k_l-epoch=18-acc=0.923.ckpt'
# model_path = 'saved_training/us+esc50_8k_l-epoch=17-acc=0.920.ckpt'

ckpt = torch.load(model_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

# file_path = '/home/super/datasets-nas/ESC-50/audio_32k/2-82367-A-10.wav'
file_path = '/home/super/datasets-nas/ESC-50/audio_32k/4-255371-A-47.wav'
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Train/audioset_-Yi8HR9alI0_280.wav' # drilling 0.9960918
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Train/audioset_P-CwFZOOTKc_30.wav' # drilling 0.9972669
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Church bell/audioset_NwfdXxNB6zs_260.wav' # street_music 0.5715629
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Church bell/audioset_JHjO_5gmDhg_30.wav' # dog_bark 0.6590187
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Drill/audioset_OuW61_qqo2A_10.wav' # drilling 0.9968809
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Drill/audioset_JKTY5v98gIk_510.wav' # drilling 0.9987525
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Dog/audioset_-YawS1V9O5U_30.wav' # dog_bark 0.9907504
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Dog/audioset_OKs3yIoO-Qw_30.wav' # dog_bark 0.99443287
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Police car/audioset_-u8XTnRF0pE_70.wav' # siren 0.99916744
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Police car/audioset_HQQxGJKg1iM_30.wav' # siren 0.993658
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Fire engine, fire truck/audioset_-QBo1W2w8II_30.wav' # siren 0.9993649
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Fire engine, fire truck/audioset_K9wg8ya4nu8_30.wav' # siren 0.99802446
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Jet engine/audioset_ObFke2ZOH2g_130.wav' # drilling 0.9943287
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Jet engine/audioset_4YTBkPw0ILc_30.wav' # drilling 0.9969241
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Fixed-wing-aircraft,airplane/audioset_-UqPsFYEk20_50.wav' # drilling 0.9934523
file_path = '/home/super/datasets-nas/audio_merged_dataset/esc50_urbansound_audioset/train/Police car/'
#            siren:       (1158/1244) 93.09%
#         drilling:       (  29/1244)  2.33%
#         car_horn:       (   3/1244)  0.24%
#     street_music:       (  31/1244)  2.49%
# children_playing:       (  10/1244)  0.80%
#           engine:       (   7/1244)  0.56%
#         dog_bark:       (   6/1244)  0.48%
# file_path = '/home/super/datasets-nas/audio/softech/Motosega'
#         drilling: (   3/3   ) 100.00%
# file_path = '/home/super/datasets-nas/audio/softech/Aereo'
# file_path = './examples_audio/4-255371-A-47.wav'
# file_path = './examples_audio/urban_sound_98223-7-10-0.wav'

results = {}
num_wav = 0

def inference(in_val):
    result = model.inference(in_val)
    win_classes = np.argmax(result['clipwise_output'], axis=1)
    win_class_index = win_classes[0]
    win_class_name = class_mapping[str(win_class_index)]
    win_class_prob = result['clipwise_output'][0][win_class_index]
    return win_class_name, win_class_prob


def predict(file_path):
    global num_wav 
    y, sr = librosa.load(file_path, sr=None)
    #print(f"audio len: {len(y)}; sampling rate: {sr}")
    if len(y) < 100:
        print(f"Input signal length={len(y)} is too small")
        return
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    #print(f"audio len: {len(y)}; sampling rate: 32000")
    # se il file audio è troppo lungo (len(y) > 300000) da errore nella trasformazione del audio in immagine
    # File "/home/super/nic/HTS-Audio-Transformer/model/htsat.py", line 781, in forward
    # x = self.reshape_wav2img(x)
    # RuntimeError: Input and output sizes should be greater than 0, but got input (H: 0, W: 64) output (H: 1024, W: 64)
    print(file_path)
    if len(y) > max_audio_len:
        for start in range(0, len(y), max_audio_len):
            in_val = np.array([y[start:start+max_audio_len]])
            win_class_name, win_class_prob = inference(in_val)

            print(win_class_name, win_class_prob)
            results[win_class_name] = results.get(win_class_name, 0) + 1
            num_wav += 1
    else:
        in_val = np.array([y])
        win_class_name, win_class_prob = inference(in_val)
        print(win_class_name, win_class_prob)
        results[win_class_name] = results.get(win_class_name, 0) + 1
        num_wav += 1


if os.path.isfile(file_path):
    predict(file_path)
elif os.path.isdir(file_path):
    for f in os.listdir(file_path):
        predict(os.path.join(file_path, f))
    print(file_path)
    for key, value in results.items():
        print(f"{key:>20}: ({value:>4}/{num_wav:<4}) {value/num_wav*100:3.2f}%")