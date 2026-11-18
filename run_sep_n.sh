#!/bin/bash
# run_sep_n.sh <tag> <seed> <lambda> <mu> <radius> <n_agents>
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
tag=$1; seed=$2; lam=$3; mu=$4; rad=$5; n=$6
export MARVEL_COORD_OVERLAP_WEIGHT=$lam
export MARVEL_COORD_MAX_DETOUR_RATIO=1.1
export MARVEL_COORD_TABU_WAYPOINT_STEPS=2
export MARVEL_COORD_SEPARATION_WEIGHT=$mu
export MARVEL_COORD_SEPARATION_RADIUS=$rad
export MARVEL_COORD_SEPARATION_REDUCE=min
export OMP_NUM_THREADS=1
conda run -n marvel_repro python run_overlap_experiment.py \
  --mode full --episodes 100 --workers 8 --n-agents "$n" --fov 120 --sensor-range 10 \
  --seed "$seed" --output "results/overlap_coord/sep/${tag}.jsonl"
