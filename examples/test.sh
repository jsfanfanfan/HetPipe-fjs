#!/bin/bash
# bash sbatch.sh 2 2 test.sh 
# module load anaconda/5.3.1_py37
source activate /gf3/home/wjliu/.conda/envs/cu113/

export CUDA=/gf3/software/cuda-11.3
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH

# CPUTI
export LD_LIBRARY_PATH=$CUDA/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# nccl
export NCCL=$CUDA

# nccl for horovod
# export HOROVOD_NCCL_HOME=$NCCL

# nccl for tf
# export TF_NCCL_VERSION='2.8.4'
# export NCCL_INSTALL_PATH=$NCCL

GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=192.180.15.60
MASTER_PORT=6010
NNODES=2
NODE_RANK=$1 # node rank 一般怎么设定
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DATA_PATH=/ssd/datasets/openwebtext/my_gpt2_text_document
VOCAB_FILE=/ssd/datasets/openwebtext/gpt2-vocab.json
MERGE_FILE=/ssd/datasets/openwebtext/gpt2-merges.txt
CHECKPOINT_PATH=/gf3/home/dyb/checkpoints/deepspeed/gpt

# ip_list="192.180.15.60,192.180.15.70"
#        --checkpoint-activations \  
#        --train-iters 500000 \

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
# DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --master_addr $MASTER_ADDR --master_port $MASTER_PORT"


# mpirun --np 8 --hostfile "./examples/hostfile"  --allow-run-as-root 
python -um torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_gpt.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --num-layers-per-virtual-pipeline-stage 3 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size 4 \
       --global-batch-size 32 \
       --seq-length 1024 \
       --max-position-embeddings 1024 \
       --train-iters 10 \
       --lr-decay-iters 320000 \
       --save $CHECKPOINT_PATH \
       --load $CHECKPOINT_PATH \
       --data-path $DATA_PATH \
       --vocab-file $VOCAB_FILE \
       --merge-file $MERGE_FILE \
       --data-impl mmap \
       --split 949,50,1 \
       --distributed-backend nccl \
       --lr 0.00015 \
       --min-lr 1.0e-5 \
       --lr-decay-style cosine \
       --weight-decay 1e-2 \
       --clip-grad 1.0 \
       --lr-warmup-fraction .01 \
       --log-interval 10 \
       --save-interval 10000 \
       --eval-interval 10000 \
       --eval-iters 10 \
       --checkpoint-activations \
