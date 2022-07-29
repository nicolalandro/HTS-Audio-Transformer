source venv/bin/activate

for i in 6 7 8 9
do
    MASTER_ADDR=localhost MASTER_PORT=29502 FOLD=$i\
    ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=2 \
    python main.py train > "test_urbansound32k_fold${i}_from_scratch.log"
done