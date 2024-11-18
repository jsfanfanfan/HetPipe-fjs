
#!/bin/bash

# module load anaconda/5.3.1_py37
source activate /dat/dyb/anaconda3/envs/cu113

export CUDA=/gf3/software/cuda-11.3
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH

# CPUTI
export LD_LIBRARY_PATH=$CUDA/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# nccl
export NCCL=$CUDA
GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=192.180.15.60
MASTER_PORT=6042
NNODES=${2:-"1"}
NODE_RANK=${1:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

task_name=${3:-"test"}
echo $task_name
# DATA_PATH=/gf3/home/dyb/code/deepspeed/Megatron-LM/subtrain.json
DATA_PATH=/dat/dyb/data/Bert_Dataset/SQuad/train-v1.1.json
VOCAB_FILE=/gf3/home/dyb/code/deepspeed/Megatron-LM/vocab
CHECKPOINT_PATH=/dat/dyb/checkpoints/deepspeed/Squad
CACHE_DIR=/dat/dyb/data/Squad_cache
LOG_DIR=/dat/dyb/logs/hete_pipe/Squad_Bert
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       run_bert_on_squad.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 2048 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 16 \
       --seq-length 512 \
       --max-position-embeddings 512 \
       --train-iters 15 \
       --save $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.0001 \
       --lr-decay-style linear \
       --min-lr 1.0e-5 \
       --lr-decay-iters 990000 \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 100 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10 \
       --cache-dir $CACHE_DIR \
       --accumulate-allreduce-grads-in-fp32 \
       --attention-softmax-in-fp32  \
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --task_name $task_name \
       --log_dir $LOG_DIR \
 