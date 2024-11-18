#!/bin/bash
# bash sbatch.sh gn[34-35] 2 pretrain_bert_distributed_with_pmp.sh
echo "使用方法："
echo "bash ./sbatch.sh 2或gn[1-2] 进程数 shell脚本"
echo "Example: bash sbatch.sh gn0 4 wait.sh 或 bash sbatch.sh 1 4 wait.sh"



if [[ "$1" -gt 0 ]] 2>/dev/null; then
    if [[ "$3" =~ "torch_dist.sh" ]]; then
        sbatch -N $1 -n $1 --ntasks-per-node=1 ./srun.sh $1
    else
        sbatch -N $1 -n $2 $3
    fi
elif [[ "$1" =~ "gn" ]]; then
    if [[ "$3" =~ "pretrain_bert_distributed_with_pmp.sh" ]]; then
        nnodes=`sinfo --node $1 |awk '{print $4}'|sed -n '2p'`
        sbatch -w $1 -n $nnodes --ntasks-per-node=1 -J bert_large ./srun.sh $nnodes 
    else
        sbatch -w $1 -n $2 $3
    fi
else
    echo "请输入正确的节点数或者节点名"
fi
# --help    # 显示帮助信息；
# -J, --job-name=<jobname>    # 指定该作业的作业名；
# -n, --ntasks=<number>    # 任务数，但是sbatch并不会执行任务，当需要申请相应的资源来运行脚本；
# --ntasks-per-node=<ntasks>    # 每个节点的任务数，--ntasks参数的优先级高于该参数，如果使用--ntasks这个参数，那么将会变为每个节点最多运行的任务数；
# -o, --output=<filename pattern>    # 输出文件，作业脚本中的输出将会输出到该文件；
# -p, --partition=<partition_names>    # 将作业提交到对应分区；
# -w, --nodelist=<node name list>  # 指定申请的节点；
# -x, --exclude=<node name list>   # 排除指定的节点；
