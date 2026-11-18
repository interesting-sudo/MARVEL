#!/bin/bash
# v5 批次：8 机低 mu、分离项时间窗（horizon）、4 机种子复现、2 机低 mu
cd /home/dingzhili2026/research_ws/MARVEL || exit 1
export PATH=/home/dingzhili2026/miniconda3/bin:$PATH
export OMP_NUM_THREADS=1
OUT=results/overlap_coord/sep

run() {
  tag=$1; seed=$2; n=$3; mode=$4; lam=$5; mu=$6; rad=$7; hor=$8; eps=$9; wk=${10}
  echo "[$(date +%H:%M:%S)] START $tag n=$n mode=$mode lam=$lam mu=$mu horizon=$hor"
  MARVEL_COORD_OVERLAP_WEIGHT=$lam \
  MARVEL_COORD_MAX_DETOUR_RATIO=1.1 \
  MARVEL_COORD_TABU_WAYPOINT_STEPS=2 \
  MARVEL_COORD_SEPARATION_WEIGHT=$mu \
  MARVEL_COORD_SEPARATION_RADIUS=$rad \
  MARVEL_COORD_SEPARATION_REDUCE=min \
  MARVEL_COORD_SEPARATION_HORIZON=$hor \
  conda run -n marvel_repro python run_overlap_experiment.py \
    --mode "$mode" --episodes "$eps" --workers "$wk" --n-agents "$n" \
    --fov 120 --sensor-range 10 --seed "$seed" \
    --output "$OUT/$tag.jsonl" > "$OUT/$tag.log" 2>&1
  echo "[$(date +%H:%M:%S)] DONE $tag"
}

run a8_mu100   20261118 8 full 40 100 25 inf 100 9 &
run a4_mu100_s2 20261119 4 full 40 100 25 inf 100 7 &
run h10 20261118 8 full 40 200 25 10 100 9 &
run h20 20261118 8 full 40 200 25 20 100 9 &
run h30 20261118 8 full 40 200 25 30 100 9 &
run a2_mu100 20261118 2 full 40 100 25 inf 100 6 &
wait
echo "[$(date +%H:%M:%S)] ALL DONE"
