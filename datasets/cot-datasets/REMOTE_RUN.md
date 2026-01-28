Remote run instructions (bash)
=============================

This document explains simple ways to run `run_qwen3_competition_math.py` on a remote host and disconnect your laptop while the job keeps running.

Prerequisites
- Script is in `/csproject/t3_sjchenaa/reasoning-embedding/Qwen3-32B-COT` on the remote.
- Put API credentials in the repo root `.env` on the remote (keys: `DASHSCOPE_API_KEY` or `BAILIAN_API_KEY`, and `DASHSCOPE_BASE_URL`/`BAILIAN_API_ENDPOINT`).
- If you use conda, make sure the `ai` env exists and contains `openai` and `datasets`.

Quick: start in tmux (recommended)
```bash
ssh user@remote.host
cd /csproject/t3_sjchenaa/reasoning-embedding/Qwen3-32B-COT
# (optional) prepare conda
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai
# use the helper to start in tmux
./run_remote.sh tmux
# detach (if attached) with Ctrl-b d, or reattach:
# tmux attach -t qwenrun
# watch JSONL output
tail -f qwen3_32b_competition_math_outputs.jsonl
```

No fuss: nohup (no interactive reattach)
```bash
ssh user@remote.host
cd /csproject/t3_sjchenaa/reasoning-embedding/Qwen3-32B-COT
./run_remote.sh nohup
# follow logs
tail -f run_remote.log
```

Systemd (for production / auto-restart)
1. Copy `deploy/qwen3_runner.service` to `/etc/systemd/system/qwen3_runner.service` and update `User` and paths.
2. Reload systemd and enable the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable --now qwen3_runner.service
sudo journalctl -u qwen3_runner -f
```

Other tips
- To resume or skip already-done problems, inspect `qwen3_32b_competition_math_outputs.jsonl` for logged `question_id` values.
- If you want absolutely no terminal output, the runner already writes JSONL only; `run_remote.sh` redirects stdout/stderr to `run_remote.log` by default.

If you want, I can:
- Add a resume helper that reads the JSONL and continues from the last logged `question_id`.
- Make `run_remote.sh` executable for you (it already is created; run `chmod +x run_remote.sh`).

