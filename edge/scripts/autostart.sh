# Sourced from ~/.bashrc (one line added by the deploy, week 6): opening the Termux app starts the inspector and
# opens Chrome on it - but only if it isn't already running, so extra Termux sessions don't re-open Chrome.
# Disable: delete the "bharatai autostart" line from ~/.bashrc (or: touch ~/bharatai/no_autostart).
if [ ! -f ~/bharatai/no_autostart ] && ! pgrep -f "[p]ython -m app.main" >/dev/null 2>&1; then
  sh ~/bharatai/edge/scripts/start_inspector.sh
fi
