# Remote MLflow Server Setup Guide

This guide documents the complete setup process for a remote MLflow tracking server on AWS EC2 with S3 artifact storage and authentication.

## Quick Start

### On EC2 Instance

1. **Copy the setup script to your EC2 instance:**
   ```bash
   scp scripts/setup_remote_mlflow_server.sh ubuntu@your-ec2-instance:~/
   ```

2. **SSH into your EC2 instance:**
   ```bash
   ssh ubuntu@your-ec2-instance
   ```

3. **Run the setup script:**
   ```bash
   chmod +x setup_remote_mlflow_server.sh
   ./setup_remote_mlflow_server.sh
   ```

4. **Follow the interactive prompts to configure:**
   - AWS credentials
   - MLflow username/password
   - Systemd service (optional)

### On Your Local Machine

1. **Set up SSH tunnel:**
   ```bash
   ssh -L 5001:localhost:5000 ubuntu@your-ec2-instance
   ```

2. **Set authentication credentials:**
   ```bash
   export MLFLOW_TRACKING_USERNAME="admin"
   export MLFLOW_TRACKING_PASSWORD="your-password"
   ```

3. **Test the connection:**
   ```bash
   python scripts/test_remote_mlflow.py
   ```

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    Local Development Machine                 │
│                                                              │
│  ┌──────────────────┐     SSH Tunnel (port 5001)           │
│  │  Training Script │────────────────────────┐              │
│  │  or Test Script  │                        │              │
│  └──────────────────┘                        │              │
└──────────────────────────────────────────────┼──────────────┘
                                               │
                                               ▼
┌─────────────────────────────────────────────────────────────┐
│                      EC2 Instance                            │
│                                                              │
│  ┌───────────────────────────────────────────────┐          │
│  │         MLflow Server (port 5000)             │          │
│  │  - Basic Auth (username/password)             │          │
│  │  - Backend: SQLite (mlflow.db)                │          │
│  │  - Artifacts: S3 proxy mode                   │          │
│  └───────────┬───────────────────┬───────────────┘          │
│              │                   │                           │
│              ▼                   ▼                           │
│        ┌──────────┐        ┌──────────┐                     │
│        │mlflow.db │        │~/.aws/   │                     │
│        │(metadata)│        │credentials│                    │
│        └──────────┘        └──────────┘                     │
└─────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                        ┌───────────────────────┐
                        │   AWS S3 Bucket       │
                        │   basketworld/        │
                        │   mlflow-artifacts/   │
                        └───────────────────────┘
```

## What the Setup Script Does

### 1. System Dependencies
- Updates package lists
- Installs Python 3, pip, and venv
- Installs AWS CLI v2
- Installs useful tools (htop, tmux, git, curl)

### 2. Installation Directory
- Creates `~/mlflow-server/` directory
- Sets up Python virtual environment
- Installs MLflow and boto3

### 3. AWS Configuration
- Creates `~/.aws/credentials` with your AWS credentials
- Creates `~/.aws/config` with region settings
- Verifies S3 bucket access

### 4. MLflow Server Configuration
- Creates startup script with proper flags
- Configures S3 artifact storage with `--artifacts-destination` (proxy mode)
- Sets up SQLite backend for metadata
- Enables basic authentication with `--app-name basic-auth`

### 5. Optional Systemd Service
- Creates systemd service for automatic startup
- Configures service to restart on failure
- Enables logging via journalctl

### 6. Authentication Setup
- Creates `.mlflow_auth_config` file with credentials
- Sets up username/password for client access

## Configuration Details

### Server Flags Explained

```bash
mlflow server \
    --backend-store-uri "sqlite:///mlflow.db" \        # Metadata storage
    --artifacts-destination "s3://bucket/path" \       # S3 artifacts (proxy mode)
    --host "0.0.0.0" \                                 # Listen on all interfaces
    --port 5000 \                                      # Server port
    --app-name basic-auth                              # Enable authentication
```

**Key Points:**

1. **`--artifacts-destination` vs `--default-artifact-root`:**
   - `--artifacts-destination` → New runs get `mlflow-artifacts:/` URIs (server proxies S3)
   - `--default-artifact-root` → Old method, runs get `s3://` URIs (client needs credentials)
   - **Always use** `--artifacts-destination` for new setups

2. **Backend Store:**
   - SQLite: Simple, single-file database (fine for small teams)
   - PostgreSQL/MySQL: For production with multiple users

3. **Authentication:**
   - `--app-name basic-auth` enables username/password
   - Clients must set `MLFLOW_TRACKING_USERNAME` and `MLFLOW_TRACKING_PASSWORD`

## Server Management

### Manual Start/Stop

Start the server manually:
```bash
cd ~/mlflow-server
./start_mlflow_server.sh
```

Stop the server (if running in foreground):
```bash
Ctrl+C
```

Run in background with nohup:
```bash
cd ~/mlflow-server
nohup ./start_mlflow_server.sh > mlflow.log 2>&1 &
echo $! > mlflow.pid  # Save process ID
```

Stop background process:
```bash
kill $(cat ~/mlflow-server/mlflow.pid)
```

### Systemd Service (If Configured)

Start:
```bash
sudo systemctl start mlflow-server
```

Stop:
```bash
sudo systemctl stop mlflow-server
```

Restart:
```bash
sudo systemctl restart mlflow-server
```

Check status:
```bash
sudo systemctl status mlflow-server
```

View logs:
```bash
sudo journalctl -u mlflow-server -f
```

Enable auto-start on boot:
```bash
sudo systemctl enable mlflow-server
```

## Client Configuration

### Environment Variables

Set these on your local machine:
```bash
export MLFLOW_TRACKING_USERNAME="admin"
export MLFLOW_TRACKING_PASSWORD="your-password"
```

### SSH Tunnel

Create SSH tunnel (keeps terminal open):
```bash
ssh -L 5001:localhost:5000 ubuntu@your-ec2-instance
```

Or run in background:
```bash
ssh -f -N -L 5001:localhost:5000 ubuntu@your-ec2-instance
```

Kill background tunnel:
```bash
ps aux | grep "ssh -f -N -L 5001"
kill <pid>
```

### Python Client

```python
import mlflow
import os

# Set authentication
os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "your-password"

# Connect to server (via SSH tunnel)
mlflow.set_tracking_uri("http://localhost:5001")

# Test connection
experiments = mlflow.search_experiments()
print(f"Connected! Found {len(experiments)} experiments")

# Use as normal
mlflow.set_experiment("my-experiment")
with mlflow.start_run():
    mlflow.log_param("param", "value")
    mlflow.log_metric("metric", 1.0)
```

## Troubleshooting

### Server Won't Start

**Check if port is already in use:**
```bash
sudo netstat -tlnp | grep 5000
# or
sudo lsof -i :5000
```

**Kill existing process:**
```bash
sudo kill -9 <pid>
```

### Can't Connect from Local Machine

**Check server is running:**
```bash
# On EC2
curl http://localhost:5000/health
```

**Check SSH tunnel:**
```bash
# On local machine
curl http://localhost:5001/health
```

**Check security group:**
- If not using SSH tunnel, ensure EC2 security group allows inbound on port 5000

### Authentication Errors

**Verify credentials are set:**
```bash
echo $MLFLOW_TRACKING_USERNAME
echo $MLFLOW_TRACKING_PASSWORD
```

**Check server authentication config:**
```bash
# On EC2
cat ~/mlflow-server/.mlflow_auth_config
```

### S3 Access Errors

**Verify AWS credentials:**
```bash
# On EC2
cat ~/.aws/credentials
aws s3 ls s3://basketworld/
```

**Check S3 bucket permissions:**
- Ensure IAM user/role has `s3:GetObject`, `s3:PutObject`, `s3:ListBucket`

**Common error: "Partial credentials found"**
- Make sure both `aws_access_key_id` and `aws_secret_access_key` are set
- Check that profile name matches in credentials and server startup script

### Artifact Retrieval Fails

**Check artifact URI format:**
```python
run = mlflow.get_run(run_id)
print(run.info.artifact_uri)
```

- Should be `mlflow-artifacts:/...` (good - server proxies)
- If `s3://...` (old - requires client credentials)

**Solution for old runs:**
Set AWS credentials on client:
```bash
export AWS_PROFILE=basketworld
# or
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
```

**Solution for new runs:**
Restart server with `--artifacts-destination` flag (already in setup script)

## Security Considerations

### Production Recommendations

1. **Use HTTPS:**
   - Put MLflow behind nginx with SSL certificate
   - Use Let's Encrypt for free SSL

2. **Use SSH Tunnel:**
   - Don't expose MLflow port directly to internet
   - Always use SSH tunnel: `ssh -L 5001:localhost:5000 user@host`

3. **Strong Credentials:**
   - Use strong, unique passwords
   - Consider implementing token-based auth

4. **Restrict Security Group:**
   - Only allow SSH (port 22) from your IP
   - Don't open port 5000 to 0.0.0.0/0

5. **S3 Bucket Permissions:**
   - Use IAM roles instead of access keys when possible
   - Apply least-privilege principle
   - Enable S3 bucket versioning and encryption

6. **Regular Backups:**
   - Back up `mlflow.db` regularly
   - S3 bucket should have versioning enabled

### Backup Script

```bash
#!/bin/bash
# Backup MLflow database to S3
BACKUP_FILE="mlflow.db.backup.$(date +%Y%m%d-%H%M%S)"
cd ~/mlflow-server
cp mlflow.db "$BACKUP_FILE"
aws s3 cp "$BACKUP_FILE" s3://basketworld/mlflow-backups/
rm "$BACKUP_FILE"
echo "Backup complete: s3://basketworld/mlflow-backups/$BACKUP_FILE"
```

## Testing

Use the provided test script to verify everything works:

```bash
# On local machine with SSH tunnel active
python scripts/test_remote_mlflow.py
```

This will:
1. ✓ Verify AWS credentials
2. ✓ Test server connectivity
3. ✓ Train a simple sklearn model
4. ✓ Log parameters, metrics, and artifacts
5. ✓ Test artifact retrieval from S3

## Files Created

After running the setup script:

```
~/mlflow-server/
├── venv/                          # Python virtual environment
├── mlflow.db                      # SQLite database (metadata)
├── start_mlflow_server.sh         # Server startup script
├── .mlflow_auth_config            # Authentication credentials
└── mlflow.log                     # Server logs (if using nohup)

~/.aws/
├── credentials                    # AWS credentials
└── config                         # AWS region config

/etc/systemd/system/
└── mlflow-server.service          # Systemd service (optional)
```

## Advanced Configuration

### Using PostgreSQL Backend

For production with multiple concurrent users:

```bash
# Install PostgreSQL
sudo apt-get install postgresql postgresql-contrib

# Create database and user
sudo -u postgres psql
CREATE DATABASE mlflow;
CREATE USER mlflow WITH PASSWORD 'password';
GRANT ALL PRIVILEGES ON DATABASE mlflow TO mlflow;

# Update server startup to use PostgreSQL
mlflow server \
    --backend-store-uri postgresql://mlflow:password@localhost/mlflow \
    --artifacts-destination s3://basketworld/mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000 \
    --app-name basic-auth
```

### Multiple Environments

Create separate tracking servers for dev/staging/prod:

```bash
# Development (port 5000)
mlflow server --port 5000 --artifacts-destination s3://bucket/dev/

# Staging (port 5001)
mlflow server --port 5001 --artifacts-destination s3://bucket/staging/

# Production (port 5002)
mlflow server --port 5002 --artifacts-destination s3://bucket/prod/
```

### Model Registry

Enable model registry (requires backend database):

```python
import mlflow

# Log model with registry
mlflow.sklearn.log_model(
    model,
    "model",
    registered_model_name="my_model"
)

# Transition model stage
client = mlflow.tracking.MlflowClient()
client.transition_model_version_stage(
    name="my_model",
    version=1,
    stage="Production"
)
```

## Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLflow Tracking](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Authentication](https://mlflow.org/docs/latest/auth/index.html)
- [AWS S3 Documentation](https://docs.aws.amazon.com/s3/)

## Support

If you encounter issues not covered here:

1. Check MLflow logs: `sudo journalctl -u mlflow-server -f`
2. Check server response: `curl http://localhost:5000/health`
3. Verify AWS credentials: `aws s3 ls s3://basketworld/`
4. Test with the test script: `python scripts/test_remote_mlflow.py`

