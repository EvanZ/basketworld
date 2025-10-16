#!/bin/bash
# Stop all MLflow processes

echo "Stopping MLflow processes..."
pkill -f 'mlflow ui'
pkill -f 'mlflow server'
pkill -f 'gunicorn.*mlflow'
sleep 2

# Check if any are still running
if ps aux | grep -v grep | grep mlflow >/dev/null ; then
    echo "⚠️  Some MLflow processes are still running:"
    ps aux | grep -v grep | grep mlflow
else
    echo "✅ All MLflow processes stopped"
fi
