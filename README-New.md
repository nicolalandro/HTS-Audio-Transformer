# How to

* download the dataset ` wget https://github.com/karoldvl/ESC-50/archive/master.zip`
* install sox `sudo apt install sox` (and python)
* install python requirements: `pip install -r requirements.txt`
* run jupyter notebook at esc-50/prep_esc50.ipynb
* change info into config.py
    *
* install ffmpeg `sudo apt install ffmpeg`
* downgrade protobuf `pip install protobuf==3.20`
* run train 
```
MASTER_ADDR=localhost MASTER_PORT=29500 ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python main.py train
```
* run test
```
MASTER_ADDR=localhost MASTER_PORT=29500 ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python main.py test
```
* run demo ```python demo.py```
* gradio app ```python demo_app.py```