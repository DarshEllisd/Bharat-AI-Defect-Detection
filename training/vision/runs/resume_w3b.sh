#!/usr/bin/env bash
# Resume week-3 training (GPU only, one job at a time). Safe to re-run: finished steps are skipped.
#  1. train_all.sh   - remaining EfficientAD models (mvtec_carpet)
#  2. after_w3b.sh   - re-evaluate all 4 with the aligned heatmap (P-50); own_fabric_s18 (P-51)
#  3. YOLO11n retrain on the merged data incl. session 6 (first model kept as yolo11n_combined_v1)
#   wsl -e bash /mnt/c/Users/Darsh/Desktop/BharatAI-SOC/training/vision/runs/resume_w3b.sh
D=/mnt/c/Users/Darsh/Desktop/BharatAI-SOC/training/vision/runs
L=$HOME/bharatai-build/models/efficientad/logs
Y=$HOME/bharatai-build/models/yolo
bash $D/train_all.sh && bash $D/after_w3b.sh >> $L/after_w3b.log 2>&1
if [ ! -f $Y/yolo11n_combined/.session6 ]; then
  [ -d $Y/yolo11n_combined ] && [ ! -d $Y/yolo11n_combined_v1 ] && mv $Y/yolo11n_combined $Y/yolo11n_combined_v1
  bash $D/train_yolo.sh && touch $Y/yolo11n_combined/.session6
fi
echo RESUME_DONE
