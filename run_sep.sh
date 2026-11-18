#!/bin/bash
# 通用运行器：run_sep.sh <tag> <seed> <lambda> <mu> <radius> [extra env...]
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
tag=$1; seed=$2; lam=$3; mu=$4; rad=$5; shift 5
export MARVEL_COORD_OVERLAP_WEIGHT=$lam
export MARVEL_COORD_MAX_DETOUR_RATIO=1.1
export MARVEL_COORD_TABU_WAYPOINT_STEPS=2
export MARVEL_COORD_SEPARATION_WEIGHT=$mu
export MARVEL_COORD_SEPARATION_RADIUS=$rad
export MARVEL_COORD_SEPARATION_REDUCE=min
for kv in "$@"; do export "$kv"; done
export OMP_NUM_THREADS=1
conda run -n marvel_repro python run_overlap_experiment.py \
  --mode full --episodes 100 --workers 8 --n-agents 8 --fov 120 --sensor-range 10 \
  --seed "$seed" --output "results/overlap_coord/sep/${tag}.jsonl"
