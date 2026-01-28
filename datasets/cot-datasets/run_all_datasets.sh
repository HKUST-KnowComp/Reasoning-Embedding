#!/bin/tcsh
# Parallel runner for 3 datasets with shared API key
# Each process runs in the background with its own log file

set SCRIPT_DIR = `dirname $0`
cd $SCRIPT_DIR

# Create logs directory if it doesn't exist
mkdir -p logs

# Use full path to python to ensure conda environment is used
set PYTHON = /csproject/t3_sjchenaa/miniconda3/envs/ai/bin/python

echo "Starting 3 dataset processes in parallel..."
echo "Using shared DASHSCOPE_API_KEY for all processes"
echo "Python: $PYTHON"
echo "Started at: `date`"
echo ""

# Start each dataset in background
nohup $PYTHON run_dataset.py math500 --limit 500 >& logs/math500.log &
set PID1 = $!
echo "Started math500 - 500 problems (PID: $PID1)"

nohup $PYTHON run_dataset.py numina_math --limit 200 >& logs/numina_math.log &
set PID2 = $!
echo "Started numina_math - 200 problems (PID: $PID2)"

nohup $PYTHON run_dataset.py gsm8k --limit 500 >& logs/gsm8k.log &
set PID3 = $!
echo "Started gsm8k - 500 problems (PID: $PID3)"

echo ""
echo "All processes started!"
echo ""
echo "Process IDs:"
echo "  math500:      $PID1"
echo "  numina_math:  $PID2"
echo "  gsm8k:        $PID3"
echo ""
echo "Outputs: outputs/"
echo "Logs:    logs/"
echo ""
echo "Monitor progress:"
echo "  tail -f logs/math500.log"
echo "  tail -f logs/numina_math.log"
echo "  tail -f logs/gsm8k.log"
echo ""
echo "Check running processes:"
echo "  ps aux | grep run_dataset.py"
echo ""
echo "Kill all:"
echo "  kill $PID1 $PID2 $PID3"
