#!/bin/bash
# v7：R=15 的稳定性与邻域确认
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
OUT=results/overlap_coord/sep

run() {
  tag=$1; seed=$2; n=$3; lam=$4; mu=$5; rad=$6; eps=$7; wk=$8
  echo "[$(date +%H:%M:%S)] START $tag n=$n lam=$lam mu=$mu R=$rad seed=$seed"
  MARVEL_COORD_OVERLAP_WEIGHT=$lam \
  MARVEL_COORD_MAX_DETOUR_RATIO=1.1 \
  MARVEL_COORD_TABU_WAYPOINT_STEPS=2 \
  MARVEL_COORD_SEPARATION_WEIGHT=$mu \
  MARVEL_COORD_SEPARATION_RADIUS=$rad \
  MARVEL_COORD_SEPARATION_REDUCE=min \
  conda run -n marvel_repro python run_overlap_experiment.py \
    --mode full --episodes "$eps" --workers "$wk" --n-agents "$n" \
    --fov 120 --sensor-range 10 --seed "$seed" \
    --output "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $tag"
}

run a8_r15_s2   20261119 8 40 200 15 100 10 &
run a8_r15_mu400 20261118 8 40 400 15 100 10 &
run a8_r12      20261118 8 40 200 12 100 10 &
run a8_r18      20261118 8 40 200 18 100 10 &
run a4_r20_s2   20261119 4 40 100 20 100 8 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
