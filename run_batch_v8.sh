#!/bin/bash
# v8：统一参数 (mu=200, R=15) 在 4/2 机的表现 + R15 与 horizon 的联合
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
OUT=results/overlap_coord/sep

run() {
  tag=$1; seed=$2; n=$3; mu=$4; rad=$5; hor=$6; eps=$7; wk=$8
  echo "[$(date +%H:%M:%S)] START $tag n=$n mu=$mu R=$rad hor=$hor"
  MARVEL_COORD_OVERLAP_WEIGHT=40 \
  MARVEL_COORD_MAX_DETOUR_RATIO=1.1 \
  MARVEL_COORD_TABU_WAYPOINT_STEPS=2 \
  MARVEL_COORD_SEPARATION_WEIGHT=$mu \
  MARVEL_COORD_SEPARATION_RADIUS=$rad \
  MARVEL_COORD_SEPARATION_REDUCE=min \
  MARVEL_COORD_SEPARATION_HORIZON=$hor \
  conda run -n marvel_repro python run_overlap_experiment.py \
    --mode full --episodes "$eps" --workers "$wk" --n-agents "$n" \
    --fov 120 --sensor-range 10 --seed "$seed" \
    --output "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $tag"
}

run a4_r15      20261118 4 200 15 inf 100 8 &
run a2_r15      20261118 2 200 15 inf 100 6 &
run a4_r20_mu200 20261118 4 200 20 inf 100 8 &
run a8_r15_h20  20261118 8 200 15 20 100 10 &
run a8_r15_h40  20261118 8 200 15 40 100 10 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
