# MLflow Remote Server Quick Reference

Quick reference for working with the remote MLflow server on EC2.

## Initial Setup (One-time)

### On EC2 Instance

```bash
# 1. Copy and run setup script
scp scripts/setup_remote_mlflow_server.sh ubuntu@your-ec2-instance:~/
ssh ubuntu@your-ec2-instance
./setup_remote_mlflow_server.sh
```

## Daily Workflow

### Step 1: Start SSH Tunnel

```bash
# Open terminal and run:
ssh -L 5001:localhost:5000 ubuntu@your-ec2-instance

# Or run in background:
ssh -f -N -L 5001:localhost:5000 ubuntu@your-ec2-instance
```

### Step 2: Set Credentials

```bash
# Add to ~/.bashrc or ~/.zshrc for persistence:
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="your-password"

# Or set for current session:
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="your-password"
```

### Step 3: Use MLflow

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5001")
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("param", "value")
    mlflow.log_metric("metric", 1.0)
```

## Common Commands

### Server Management (On EC2)

```bash
# Start server (manual)
cd ~/mlflow-server && ./start_mlflow_server.sh

# Start with systemd
sudo systemctl start mlflow-server

# Stop with systemd
sudo systemctl stop mlflow-server

# Check status
sudo systemctl status mlflow-server

# View logs
sudo journalctl -u mlflow-server -f
```

### Testing Connection

```bash
# Test from EC2
curl http://localhost:5000/health

# Test from local (with tunnel)
curl http://localhost:5001/health

# Full test with script
python scripts/test_remote_mlflow.py
```

### Troubleshooting

```bash
# Check if server is running
ssh ubuntu@your-ec2-instance "ps aux | grep mlflow"

# Check port
ssh ubuntu@your-ec2-instance "sudo netstat -tlnp | grep 5000"

# View logs
ssh ubuntu@your-ec2-instance "sudo journalctl -u mlflow-server -n 50"

# Restart server
ssh ubuntu@your-ec2-instance "sudo systemctl restart mlflow-server"
```

## File Locations (On EC2)

```
~/mlflow-server/
├── start_mlflow_server.sh      # Startup script
├── mlflow.db                    # Database
├── .mlflow_auth_config          # Credentials
└── venv/                        # Python environment

~/.aws/
├── credentials                  # AWS credentials
└── config                       # AWS config
```

## Quick Fixes

### "Connection refused"
- Check SSH tunnel is active
- Verify server is running on EC2

### "Authentication failed"
- Check `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD` are set
- Verify credentials match server config

### "Partial credentials found"
- AWS credentials issue (already fixed in test script)
- Ensure `~/.aws/credentials` has complete profile

### Server won't start
```bash
# Check if port is in use
sudo lsof -i :5000
# Kill process if needed
sudo kill -9 <pid>
```

## Environment Variables Reference

```bash
# MLflow authentication
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="your-password"

# AWS (for local client - automatically handled by test script)
export AWS_PROFILE="basketworld"
# or
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export AWS_DEFAULT_REGION="us-west-1"
```

## Python Code Snippets

### Basic Usage

```python
import mlflow
import os

# Set credentials
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-password"

# Connect
mlflow.set_tracking_uri("http://localhost:5001")

# Log stuff
mlflow.set_experiment("test")
with mlflow.start_run():
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_metric("accuracy", 0.95)
```

### Check Connection

```python
import mlflow
mlflow.set_tracking_uri("http://localhost:5001")
experiments = mlflow.search_experiments()
print(f"Found {len(experiments)} experiments")
```

### List Recent Runs

```python
import mlflow
client = mlflow.tracking.MlflowClient()
runs = client.search_runs(
    experiment_ids=["1"],
    max_results=5,
    order_by=["start_time DESC"]
)
for run in runs:
    print(f"{run.info.run_id}: {run.data.metrics}")
```

## URLs

- **MLflow UI (via tunnel):** http://localhost:5001
- **Health Check (via tunnel):** http://localhost:5001/health
- **API (via tunnel):** http://localhost:5001/api/2.0/mlflow/

## Tips

1. **Keep SSH tunnel open** - Use tmux/screen or run in background
2. **Set env vars in shell config** - Add exports to `~/.bashrc` or `~/.zshrc`
3. **Use systemd on EC2** - Server auto-starts on reboot
4. **Test with test script** - `python scripts/test_remote_mlflow.py` verifies everything
5. **Check logs for errors** - `sudo journalctl -u mlflow-server -f`

## See Also

- Full setup guide: `scripts/REMOTE_MLFLOW_SETUP.md`
- Test script: `scripts/test_remote_mlflow.py`
- Setup script: `scripts/setup_remote_mlflow_server.sh`

