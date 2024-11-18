#!/bin/bash

srun -N $1 -n $1 ./pretrain_bert_distributed_with_pmp.sh
