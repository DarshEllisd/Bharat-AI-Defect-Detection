#!/data/data/com.termux/files/usr/bin/sh
# Stop the inspector, its vision_server and any running advice generation.
#   sh ~/bharatai/edge/scripts/stop_inspector.sh
for pat in "[p]ython -m app.main" "[v]ision_server" "[l]lama_main"; do
  pkill -TERM -f "$pat" 2>/dev/null
done
sleep 2
for pat in "[p]ython -m app.main" "[v]ision_server" "[l]lama_main"; do
  pkill -KILL -f "$pat" 2>/dev/null
done
rm -f ~/bharatai/run/vision.sock
echo "inspector stopped"
