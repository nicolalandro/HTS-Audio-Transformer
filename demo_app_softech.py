import os
import gradio as gr
from scipy.io.wavfile import write
import config

import torch
from model.htsat import HTSAT_Swin_Transformer
from sed_model import SEDWrapper
import librosa
import numpy as np

SR = 16000
MAX_SEC = 10
max_audio_len = MAX_SEC * SR

example_path = './examples_audio'
model_path = 'saved_training/us+esc50+AS+softech_16k_l-epoch=275-acc=0.542.ckpt'

class_mapping = ["Air brake", "Air horn, truck horn", "Animal", "Applause", "Bang", "Bark", "Bus", 
"Car alarm", "Civil defense siren", "Emergency vehicle", "Explosion", "Fire engine, fire truck", 
"Fireworks", "Fixed-wing aircraft, airplane", "Gunshot, gunfire", "Hammer", "Heavy engine", "Honk", 
"Hubbub, speech noise, speech babble", "Lawn mower", "Mechanical fan", "Motorcycle", "Police car", 
"Propeller, airscrew", "Race car, auto racing", "Shout", "Tire squeal", "Traffic noise, roadway noise", 
"Train horn", "Truck", "Vehicle horn, car horn, honking", "Vibration", "Yell", "airplane", "car_horn", 
"chainsaw", "children_playing", "church_bells", "dog_bark", "drilling", "engine", "fireworks", "hand_saw", 
"helicopter", "jackhammer", "siren", "street_music", "train"]

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

ckpt = torch.load(model_path, map_location="cpu")
model.load_state_dict(ckpt["state_dict"], strict=False)

# results = {}
# num_wav = 0

# def inference(audio):
#     sr, y = audio
#     y = y/32767.0 # scipy vs librosa
#     if len(y.shape) != 1: # to mono
#         y = y[:,0]
#     y = librosa.resample(y, orig_sr=sr, target_sr=32000)
#     in_val = np.array([y])
#     result = model.inference(in_val)
#     pred = result['clipwise_output'][0]
#     return {class_mapping[i]: float(p) for i, p in enumerate(pred)}

def inference(in_val):
    result = model.inference(in_val)
    win_classes = np.argmax(result['clipwise_output'], axis=1)
    win_class_index = win_classes[0]
    win_class_name = class_mapping[win_class_index]
    win_class_prob = result['clipwise_output'][0][win_class_index]
    pred = result['clipwise_output'][0]
    all_preds = {class_mapping[i]: float(p) for i, p in enumerate(pred)}
    return win_class_name, win_class_prob, all_preds


def predict(audio):
    num_wav = 0
    results = {}
    all_results = {}
    sr, y = audio
    y = y/32767.0 # scipy vs librosa
    if len(y.shape) != 1: # to mono
        y = y[:,0]
    # y = librosa.resample(y, orig_sr=sr, target_sr=32000)
    # y, sr = librosa.load(file_path, sr=None)
    #print(f"audio len: {len(y)}; sampling rate: {sr}")
    if len(y) < 100:
        print(f"Input signal length={len(y)} is too small")
        return None, f"Input signal length={len(y)} is too small"
    y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    #print(f"audio len: {len(y)}; sampling rate: 32000")
    # se il file audio è troppo lungo (len(y) > 300000) da errore nella trasformazione del audio in immagine
    # File "/home/super/nic/HTS-Audio-Transformer/model/htsat.py", line 781, in forward
    # x = self.reshape_wav2img(x)
    # RuntimeError: Input and output sizes should be greater than 0, but got input (H: 0, W: 64) output (H: 1024, W: 64)
    # print(file_path)
    if len(y) > max_audio_len:
        for start in range(0, len(y), max_audio_len):
            in_val = np.array([y[start:start+max_audio_len]])
            if in_val.size < SR:
                print(f"Input signal length={len(in_val)} is too short")
                break
            win_class_name, win_class_prob, all_preds = inference(in_val)

            print(win_class_name, win_class_prob)
            results[win_class_name] = results.get(win_class_name, 0) + 1
            for k,v in all_preds.items():
                all_results[k] = all_results.get(k, 0) + v
            num_wav += 1
    else:
        in_val = np.array([y])
        win_class_name, win_class_prob, all_preds = inference(in_val)
        print(win_class_name, win_class_prob)
        results[win_class_name] = results.get(win_class_name, 0) + win_class_prob
        for k,v in all_preds.items():
            all_results[k] = all_results.get(k, 0) + v
        num_wav += 1
    for key, value in results.items():
        results[key] = value/num_wav
        # print(f"{key:>20}: ({value:>4}/{num_wav:<4}) {value/num_wav*100:3.2f}%")
    # return "Ok", results
    print(all_results)
    printed_all_results = [(key, value) for key, value in sorted(all_results.items(), key=lambda item: item[1], reverse=True)]
    return all_results, f"Ok, {printed_all_results[:10]}"

title = "Softech Audio Classifier Demo"
description = """
Audio classification with Transformer Neural Network. 
The model has been trained to recognize certain classes of audio sound, 
including in particular the noise of aircraft exceeding the threshold in the control units around airports.
<img src="https://airtechitaly.com/wp-content/uploads/2021/02/A_LOGO-SOFTECH-600x600.png" width="150">
"""
# article = "<p style='text-align: center'><a href='https://arxiv.org/abs/1911.13254' target='_blank'>Music Source Separation in the Waveform Domain</a> | <a href='https://github.com/facebookresearch/demucs' target='_blank'>Github Repo</a></p>"

examples = [['test.mp3']]
gr.Interface(
    predict,
    gr.inputs.Audio(type="numpy", label="Input"),
    [gr.outputs.Label(10), gr.outputs.Textbox()],
    title=title,
    description=description,
    # article=article,
    examples=[[os.path.join(example_path, f)]
              for f in os.listdir(example_path)]
).launch(enable_queue=True)
