#!/bin/bash
# v4 补测批次：4 机降 mu 复验、2 机补测、8 机 mu=150
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
OUT=results/overlap_coord/sep

run() {
  tag=$1; seed=$2; n=$3; mode=$4; lam=$5; mu=$6; rad=$7; eps=$8; wk=$9
  echo "[$(date +%H:%M:%S)] START $tag n=$n mode=$mode lam=$lam mu=$mu rad=$rad"
  MARVEL_COORD_OVERLAP_WEIGHT=$lam \
  MARVEL_COORD_MAX_DETOUR_RATIO=1.1 \
  MARVEL_COORD_TABU_WAYPOINT_STEPS=2 \
  MARVEL_COORD_SEPARATION_WEIGHT=$mu \
  MARVEL_COORD_SEPARATION_RADIUS=$rad \
  MARVEL_COORD_SEPARATION_REDUCE=min \
  conda run -n marvel_repro python run_overlap_experiment.py \
    --mode "$mode" --episodes "$eps" --workers "$wk" --n-agents "$n" \
    --fov 120 --sensor-range 10 --seed "$seed" \
    --output "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $tag"
}

run a4_mu100 20261118 4 full 40 100 25 100 8 &
run a4_mu150 20261118 4 full 40 150 25 100 8 &
run a2_off   20261118 2 off  40 200 25 100 6 &
run a2_hard  20261118 2 full 40 200 25 100 6 &
run a8_mu150 20261118 8 full 40 150 25 100 10 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
