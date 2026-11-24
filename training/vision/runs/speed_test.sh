#!/usr/bin/env bash
# Speed/CPU check of train_efficientad.py: 1500 iterations on MVTec wood, CPU + GPU sampled at 60 s.
cd /mnt/c/Users/Darsh/Desktop/BharatAI-SOC
B=$HOME/bharatai-build
rm -rf $B/models/efficientad/smoke_wood
$B/train-venv/bin/python training/vision/train_efficientad.py --name smoke_wood --iters 1500 \
  --train $B/datasets/anomaly/wood/train90/good --val $B/datasets/anomaly/wood/val/good \
  --penalty-dir $B/data/imagenette_flat --geom-aug > /tmp/speed_test.log 2>&1 &
pid=$!
sleep 60
echo "CPU% of training process: $(ps -o pcpu= -p $pid)"; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
wait $pid
cat /tmp/speed_test.log; cat $B/models/efficientad/smoke_wood/train_log.csv
