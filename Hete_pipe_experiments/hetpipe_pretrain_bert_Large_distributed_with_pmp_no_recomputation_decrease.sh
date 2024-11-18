#!/bin/bash
# ./examples/pretrain_bert_distributed_with_pmp.sh 1 2 221010_sprecompute_gn5_9 gn9 6099
# module load anaconda/5.3.1_py37
source activate /dat/dyb/anaconda3/envs/cu113

export CUDA=/dat/software/cuda-11.3
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH
# export NCCL_IB_DISABLE=1
# CPUTI
export LD_LIBRARY_PATH=$CUDA/extras/CUPTI/lib64:$LD_LIBRARY_PATH

# nccl
export NCCL=$CUDA
GPUS_PER_NODE=4
# Change for multinode config
gn=`hostname | awk -F "n" '{print int($2)}'`
case $gn in
       4)
       rank=0
       ;;
       3)
       rank=1
       ;;
       8)
       rank=2
       ;;
       *)
       rank=3
esac

NNODES=$SLURM_NNODES
Addr=${4:-"gn4"}
port=${5:-"6070"}
MASTER_ADDR=${Addr:-"gn5"}
MASTER_PORT=${port:-"6070"}
mbs=${1:-"4"}
gbs=${2:-"128"}

NODE_RANK=${rank:-"0"}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
partition_vector=${6:-"12,6,4,2"}
task_name=DP4_981112_11_6_4_3
echo $task_name
echo $MASTER_ADDR
DATA_PATH=/dat/dyb/data/Bert_Dataset/Megatron/my-bert_text_sentence/my-bert_text_sentence
VOCAB_FILE=/gf3/home/dyb/code/deepspeed/Megatron-LM/vocab
CHECKPOINT_PATH=/dat/dyb/checkpoints/deepspeed/Bert
LOG_DIR=/dat/dyb/logs/hete_pipe
if [ "${task_name}" == "test" ]; then
       LOG_DIR=/dat/dyb/logs/hete_pipe/tmp
fi
DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
       pretrain_bert.py \
       --tensor-model-parallel-size 1 \
       --pipeline-model-parallel-size 4 \
       --num-layers 24 \
       --hidden-size 1024 \
       --num-attention-heads 16 \
       --micro-batch-size $mbs \
       --global-batch-size $gbs \
       --seq-length 256 \
       --max-position-embeddings 512 \
       --train-iters 300 \
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
       --log-interval 50 \
       --save-interval 10000 \
       --eval-interval 1000 \
       --eval-iters 10  \
       --accumulate-allreduce-grads-in-fp32 \
       --attention-softmax-in-fp32  \
       --no-masked-softmax-fusion \
       --no-bias-gelu-fusion \
       --no-bias-dropout-fusion \
       --task_name $task_name \
       --log_dir $LOG_DIR \
       --partition_vector $partition_vector \
       --hetpipe True \
       --checkpoint-activations 
       # --DDP-impl torch \
       # 
       # --sprecompute True