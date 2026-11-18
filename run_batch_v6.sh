#!/bin/bash
# v6：分离半径 R 的临界值扫描（R_sensor=10 → 两圆相切距离 20，R>20 属于过度推开）
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
OUT=results/overlap_coord/sep

run() {
  tag=$1; seed=$2; n=$3; lam=$4; mu=$5; rad=$6; eps=$7; wk=$8
  echo "[$(date +%H:%M:%S)] START $tag n=$n lam=$lam mu=$mu R=$rad"
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

run a8_r20 20261118 8 40 200 20 100 10 &
run a8_r15 20261118 8 40 200 15 100 10 &
run a8_r10 20261118 8 40 200 10 100 10 &
run a8_mu100_r20 20261118 8 40 100 20 100 10 &
run a4_mu100_r20 20261118 4 40 100 20 100 8 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
