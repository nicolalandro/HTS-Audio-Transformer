# How to

* download the dataset 

` wget https://github.com/karoldvl/ESC-50/archive/master.zip`

* install sox `sudo apt install sox` (and python)
* install python requirements: 

`pip install -r requirements.txt`

* run jupyter notebook at esc-50/prep_esc50.ipynb
* change info into config.py
    *
* install ffmpeg `sudo apt install ffmpeg`
* downgrade protobuf `pip install protobuf==3.20`
* run train 
```bash
MASTER_ADDR=localhost MASTER_PORT=29500 ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python main.py train
# or with nohup
MASTER_ADDR=localhost MASTER_PORT=29500 ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 nohup python main.py train &> test2.log &
```
* run test
```bash
MASTER_ADDR=localhost MASTER_PORT=29500 ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 python main.py test
```
* run demo 

```python demo.py```

* gradio app 

```python demo_app.py```

* **Extend dataset**: add new samples to the dataset.

```bash
cd merged_dataset
# adapt the scripts before executing the following commands
python dataset_us_esc.py
python split_dataset.py
cd ../audio_folder_dataset
python prep_audio_folder_dataset.py
```

