import config

import torch
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
import librosa
import numpy as np

class_mapping = ['airplane', 'car_horn', 'chainsaw', 'children_playing', 'church_bells', 'dog_bark',
                 'drilling', 'engine', 'fireworks', 'hand_saw', 'helicopter', 'jackhammer', 'siren', 'street_music', 'train']

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
ckpt = torch.load('saved_training/us+esc50_l-epoch=18-acc=0.923.ckpt', map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

# file_path = '/home/super/datasets-nas/ESC-50/audio_32k/2-82367-A-10.wav'
file_path = '/home/super/datasets-nas/ESC-50/audio_32k/4-255371-A-47.wav'
# file_path = './examples_audio/4-255371-A-47.wav'
# file_path = './examples_audio/urban_sound_98223-7-10-0.wav'


y, sr = librosa.load(file_path, sr=None)
y = librosa.resample(y, orig_sr=sr, target_sr=32000)
in_val = np.array([y])

result = model.inference(in_val)
win_classes = np.argmax(result['clipwise_output'], axis=1)
win_class_index = win_classes[0]
win_class_name = class_mapping[win_class_index]

print(win_class_name, result['clipwise_output'][0][win_class_index])
