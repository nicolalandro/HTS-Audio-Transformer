for i in 1 2 3 4 5
do
    MASTER_ADDR=localhost MASTER_PORT=29500 FOLD=$i\
    ROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python CUDA_VISIBLE_DEVICES=0 \
    python main.py train > "test_urbansound32k_fold${i}_from_scratch.log"
done