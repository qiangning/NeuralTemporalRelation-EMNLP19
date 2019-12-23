#!/bin/bash
name_prefix=proposed_elmo
dataset=$1
# dataset=tcr
# dataset=matres
gamma=0.3
step=10
max_epoch=30
lr=0.001
weight_decay=1e-2
lstm_hid_dim=64
nn_hid_dim=64
common_sense_emb_dim=32
granularity=0.2
bigramstats_dim=2
seed=102
skiptuning=false
skiptraining=true
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export OPENMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
EXPNAME=${name_prefix}_lr${lr}_seed${seed}
echo Experiment Name $EXPNAME
echo -----------------
echo $gamma, $step, $lr, $max_epoch
echo lstm_hid_dim=$lstm_hid_dim
echo nn_hid_dim=$nn_hid_dim
echo expname=$EXPNAME
CMD="python exp_myLSTM.py --lstm_hid_dim ${lstm_hid_dim} --nn_hid_dim ${nn_hid_dim} --pos_emb_dim 32 --step_size $step --max_epoch $max_epoch --lr $lr --gamma $gamma --expname $EXPNAME --gen_output --common_sense_emb_dim $common_sense_emb_dim --granularity $granularity --bigramstats_dim $bigramstats_dim --weight_decay ${weight_decay} --testsetname ${dataset} --sd ${seed}"
if [ "$skiptuning" == "true" ]; then
    CMD="${CMD} --skiptuning"
fi
if [ "$skiptraining" == "true" ]; then
    CMD="$CMD --skiptraining"
fi
if [ "$skiptraining" == "true" ]; then
    EXPNAME=${EXPNAME}_skiptrain
fi
mkdir -p logs output
CMD="$CMD &> logs/${EXPNAME}_${dataset}.txt"
echo Running: $CMD
eval $CMD
