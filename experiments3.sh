source venv/bin/activate

for i in 0 5
do
    MASTER_ADDR=localhost MASTER_PORT=29501 FOLD=$i\
    ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=1 \
    python main.py train > "logs/urbansound_scratch/test_urbansound32k_fold${i}_from_scratch.log"
done