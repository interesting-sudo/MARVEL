#!/bin/bash
# 硬分离上界：在 Top-12 内取"离最近队友最远"的候选，检验分离项的能力上限
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export MARVEL_COORD_OVERLAP_WEIGHT=40
export MARVEL_COORD_MAX_DETOUR_RATIO=1.1
export MARVEL_COORD_TABU_WAYPOINT_STEPS=2
export MARVEL_COORD_SEPARATION_WEIGHT=200
export MARVEL_COORD_SEPARATION_RADIUS=25
export MARVEL_COORD_SEPARATION_REDUCE=min
export OMP_NUM_THREADS=1
conda run -n marvel_repro python run_overlap_experiment.py \
  --mode full --episodes 100 --workers 8 --n-agents 8 --fov 120 --sensor-range 10 \
  --seed 20261118 --output results/overlap_coord/sep/sep_hard_r25_min.jsonl
