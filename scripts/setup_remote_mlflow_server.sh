#!/bin/bash
################################################################################
# MLflow Remote Server Setup Script for EC2
################################################################################
#
# This script documents and automates the setup of a remote MLflow tracking 
# server on an EC2 instance with:
#   - S3 artifact storage
#   - Username/password authentication
#   - SQLite backend store
#
# Usage:
#   1. Copy this script to your EC2 instance
#   2. Make it executable: chmod +x setup_remote_mlflow_server.sh
#   3. Run it: ./setup_remote_mlflow_server.sh
#
# Prerequisites:
#   - EC2 instance (Ubuntu 22.04 LTS recommended)
#   - AWS credentials with S3 access
#   - Security group allowing inbound on port 5000 (or your chosen port)
#
################################################################################

set -e  # Exit on error

# Configuration variables - EDIT THESE
MLFLOW_PORT=5000
MLFLOW_HOST="0.0.0.0"  # Listen on all interfaces (use 127.0.0.1 for localhost only)
S3_BUCKET="basketworld"
S3_ARTIFACT_PATH="mlflow-artifacts"
AWS_REGION="us-west-1"
AWS_PROFILE="basketworld"

# Authentication - EDIT THESE
MLFLOW_USERNAME="admin"  # Change this!
MLFLOW_PASSWORD="changeme123"  # Change this! Use a strong password

# Installation directory
INSTALL_DIR="$HOME/mlflow-server"
DB_PATH="$INSTALL_DIR/mlflow.db"

################################################################################
# Functions
################################################################################

print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

print_step() {
    echo ""
    echo ">>> $1"
    echo ""
}

################################################################################
# Main Installation
################################################################################

print_header "MLflow Remote Server Setup"

echo "This script will set up a production MLflow tracking server with:"
echo "  - Port: $MLFLOW_PORT"
echo "  - Host: $MLFLOW_HOST"
echo "  - S3 Bucket: s3://$S3_BUCKET/$S3_ARTIFACT_PATH"
echo "  - AWS Region: $AWS_REGION"
echo "  - Installation Directory: $INSTALL_DIR"
echo ""
read -p "Continue? (y/n) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Installation cancelled."
    exit 1
fi

################################################################################
# Step 1: System Updates and Dependencies
################################################################################

print_header "Step 1: Installing System Dependencies"

print_step "Updating package lists..."
sudo apt-get update

print_step "Installing Python 3 and pip..."
sudo apt-get install -y python3 python3-pip python3-venv

print_step "Installing AWS CLI (if not already installed)..."
if ! command -v aws &> /dev/null; then
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
    sudo apt-get install -y unzip
    unzip awscliv2.zip
    sudo ./aws/install
    rm -rf aws awscliv2.zip
    echo "✓ AWS CLI installed"
else
    echo "✓ AWS CLI already installed ($(aws --version))"
fi

print_step "Installing other useful tools..."
sudo apt-get install -y htop tmux git curl

################################################################################
# Step 2: Create Installation Directory and Virtual Environment
################################################################################

print_header "Step 2: Setting Up Installation Directory"

print_step "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

print_step "Creating Python virtual environment..."
python3 -m venv venv

print_step "Activating virtual environment..."
source venv/bin/activate

print_step "Upgrading pip..."
pip install --upgrade pip

################################################################################
# Step 3: Install MLflow and Dependencies
################################################################################

print_header "Step 3: Installing MLflow and Dependencies"

print_step "Installing MLflow..."
pip install mlflow

print_step "Installing boto3 for S3 support..."
pip install boto3

print_step "Installing additional dependencies..."
pip install pymysql cryptography  # In case you want to use MySQL backend later

print_step "Verifying MLflow installation..."
mlflow --version

################################################################################
# Step 4: AWS Credentials Setup
################################################################################

print_header "Step 4: Configuring AWS Credentials"

AWS_CREDS_DIR="$HOME/.aws"
AWS_CREDS_FILE="$AWS_CREDS_DIR/credentials"
AWS_CONFIG_FILE="$AWS_CREDS_DIR/config"

print_step "Creating AWS credentials directory..."
mkdir -p "$AWS_CREDS_DIR"

if [ -f "$AWS_CREDS_FILE" ]; then
    echo "⚠️  AWS credentials file already exists at $AWS_CREDS_FILE"
    read -p "Overwrite? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Skipping AWS credentials setup."
        echo "Make sure you have valid credentials configured!"
    else
        setup_credentials=true
    fi
else
    setup_credentials=true
fi

if [ "$setup_credentials" = true ]; then
    print_step "Setting up AWS credentials..."
    echo "Enter your AWS credentials for S3 access:"
    read -p "AWS Access Key ID: " aws_access_key_id
    read -p "AWS Secret Access Key: " aws_secret_access_key
    
    # Create credentials file
    cat > "$AWS_CREDS_FILE" <<EOF
[$AWS_PROFILE]
aws_access_key_id = $aws_access_key_id
aws_secret_access_key = $aws_secret_access_key
EOF
    
    # Create config file
    cat > "$AWS_CONFIG_FILE" <<EOF
[profile $AWS_PROFILE]
region = $AWS_REGION
EOF
    
    chmod 600 "$AWS_CREDS_FILE"
    chmod 600 "$AWS_CONFIG_FILE"
    
    echo "✓ AWS credentials configured"
fi

################################################################################
# Step 5: Verify S3 Access
################################################################################

print_header "Step 5: Verifying S3 Access"

print_step "Testing S3 bucket access..."
export AWS_PROFILE="$AWS_PROFILE"

if aws s3 ls "s3://$S3_BUCKET/" &> /dev/null; then
    echo "✓ Successfully connected to S3 bucket: s3://$S3_BUCKET/"
else
    echo "❌ Failed to access S3 bucket: s3://$S3_BUCKET/"
    echo ""
    echo "Please ensure:"
    echo "  1. The bucket exists"
    echo "  2. Your AWS credentials have access to the bucket"
    echo "  3. The bucket name and region are correct"
    echo ""
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

################################################################################
# Step 6: Create MLflow Startup Script
################################################################################

print_header "Step 6: Creating MLflow Startup Script"

STARTUP_SCRIPT="$INSTALL_DIR/start_mlflow_server.sh"

print_step "Creating startup script: $STARTUP_SCRIPT"

cat > "$STARTUP_SCRIPT" <<'EOF'
#!/bin/bash
# MLflow Server Startup Script
# This script starts the MLflow tracking server with S3 artifact storage

cd "$(dirname "$0")"

# Load configuration
MLFLOW_PORT=5000
MLFLOW_HOST="0.0.0.0"
S3_BUCKET="basketworld"
S3_ARTIFACT_PATH="mlflow-artifacts"
AWS_PROFILE="basketworld"
DB_PATH="./mlflow.db"

# Activate virtual environment
source venv/bin/activate

# Set AWS profile
export AWS_PROFILE="$AWS_PROFILE"

echo "=================================="
echo "Starting MLflow Server"
echo "=================================="
echo "Port: $MLFLOW_PORT"
echo "Host: $MLFLOW_HOST"
echo "S3 Artifacts: s3://$S3_BUCKET/$S3_ARTIFACT_PATH"
echo "Backend Store: $DB_PATH"
echo "AWS Profile: $AWS_PROFILE"
echo "=================================="
echo ""

# Start MLflow server
mlflow server \
    --backend-store-uri "sqlite:///$DB_PATH" \
    --artifacts-destination "s3://$S3_BUCKET/$S3_ARTIFACT_PATH" \
    --host "$MLFLOW_HOST" \
    --port "$MLFLOW_PORT" \
    --app-name basic-auth

# Note: The --app-name basic-auth flag enables username/password authentication
# You'll need to set MLFLOW_TRACKING_USERNAME and MLFLOW_TRACKING_PASSWORD
# environment variables or configure them in the MLflow UI
EOF

# Update values in startup script
sed -i "s|MLFLOW_PORT=5000|MLFLOW_PORT=$MLFLOW_PORT|g" "$STARTUP_SCRIPT"
sed -i "s|MLFLOW_HOST=\"0.0.0.0\"|MLFLOW_HOST=\"$MLFLOW_HOST\"|g" "$STARTUP_SCRIPT"
sed -i "s|S3_BUCKET=\"basketworld\"|S3_BUCKET=\"$S3_BUCKET\"|g" "$STARTUP_SCRIPT"
sed -i "s|S3_ARTIFACT_PATH=\"mlflow-artifacts\"|S3_ARTIFACT_PATH=\"$S3_ARTIFACT_PATH\"|g" "$STARTUP_SCRIPT"
sed -i "s|AWS_PROFILE=\"basketworld\"|AWS_PROFILE=\"$AWS_PROFILE\"|g" "$STARTUP_SCRIPT"

chmod +x "$STARTUP_SCRIPT"
echo "✓ Startup script created"

################################################################################
# Step 7: Create Systemd Service (Optional)
################################################################################

print_header "Step 7: Creating Systemd Service"

echo "Would you like to create a systemd service to run MLflow automatically?"
read -p "(y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    SERVICE_FILE="/etc/systemd/system/mlflow-server.service"
    
    print_step "Creating systemd service..."
    
    sudo bash -c "cat > $SERVICE_FILE" <<EOF
[Unit]
Description=MLflow Tracking Server
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment="PATH=$INSTALL_DIR/venv/bin:/usr/local/bin:/usr/bin:/bin"
Environment="AWS_PROFILE=$AWS_PROFILE"
ExecStart=$INSTALL_DIR/venv/bin/mlflow server \\
    --backend-store-uri sqlite:///$DB_PATH \\
    --artifacts-destination s3://$S3_BUCKET/$S3_ARTIFACT_PATH \\
    --host $MLFLOW_HOST \\
    --port $MLFLOW_PORT \\
    --app-name basic-auth
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF
    
    print_step "Enabling and starting service..."
    sudo systemctl daemon-reload
    sudo systemctl enable mlflow-server
    sudo systemctl start mlflow-server
    
    echo "✓ Systemd service created and started"
    echo ""
    echo "Service management commands:"
    echo "  sudo systemctl status mlflow-server   # Check status"
    echo "  sudo systemctl stop mlflow-server     # Stop service"
    echo "  sudo systemctl start mlflow-server    # Start service"
    echo "  sudo systemctl restart mlflow-server  # Restart service"
    echo "  sudo journalctl -u mlflow-server -f   # View logs"
else
    echo "Skipped systemd service creation."
fi

################################################################################
# Step 8: Setup Authentication
################################################################################

print_header "Step 8: Setting Up Authentication"

AUTH_CONFIG_FILE="$INSTALL_DIR/.mlflow_auth_config"

print_step "Creating authentication configuration..."

cat > "$AUTH_CONFIG_FILE" <<EOF
# MLflow Authentication Configuration
# 
# These credentials are used for basic authentication with the MLflow server.
# Clients must provide these credentials when connecting.
#
# Usage in client:
#   export MLFLOW_TRACKING_USERNAME="$MLFLOW_USERNAME"
#   export MLFLOW_TRACKING_PASSWORD="$MLFLOW_PASSWORD"
#   mlflow.set_tracking_uri("http://your-server:$MLFLOW_PORT")

MLFLOW_TRACKING_USERNAME=$MLFLOW_USERNAME
MLFLOW_TRACKING_PASSWORD=$MLFLOW_PASSWORD
EOF

chmod 600 "$AUTH_CONFIG_FILE"

echo "✓ Authentication configuration saved to: $AUTH_CONFIG_FILE"
echo ""
echo "⚠️  IMPORTANT: Keep these credentials secure!"
echo "   Username: $MLFLOW_USERNAME"
echo "   Password: $MLFLOW_PASSWORD"

################################################################################
# Step 9: Security Group / Firewall Configuration
################################################################################

print_header "Step 9: Firewall Configuration"

echo "To access MLflow remotely, you need to:"
echo ""
echo "1. AWS Security Group (if using EC2):"
echo "   - Add inbound rule for TCP port $MLFLOW_PORT"
echo "   - Source: Your IP or 0.0.0.0/0 (not recommended for production)"
echo ""
echo "2. Local Firewall (if enabled):"
echo "   sudo ufw allow $MLFLOW_PORT/tcp"
echo ""
echo "3. SSH Tunnel (Recommended for secure access):"
echo "   ssh -L 5001:localhost:$MLFLOW_PORT ubuntu@your-ec2-instance"
echo "   Then connect to: http://localhost:5001"
echo ""

read -p "Would you like to open the firewall port now? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if command -v ufw &> /dev/null; then
        sudo ufw allow "$MLFLOW_PORT/tcp"
        echo "✓ Firewall rule added"
    else
        echo "⚠️  UFW not installed, skipping firewall configuration"
    fi
fi

################################################################################
# Step 10: Final Summary
################################################################################

print_header "Installation Complete!"

echo "MLflow server has been set up successfully!"
echo ""
echo "Installation Details:"
echo "  Directory: $INSTALL_DIR"
echo "  Port: $MLFLOW_PORT"
echo "  Host: $MLFLOW_HOST"
echo "  Database: $DB_PATH"
echo "  S3 Artifacts: s3://$S3_BUCKET/$S3_ARTIFACT_PATH"
echo ""
echo "Authentication:"
echo "  Username: $MLFLOW_USERNAME"
echo "  Password: $MLFLOW_PASSWORD"
echo "  Config file: $AUTH_CONFIG_FILE"
echo ""
echo "Next Steps:"
echo ""
echo "1. Start the server (if not using systemd):"
echo "   cd $INSTALL_DIR"
echo "   ./start_mlflow_server.sh"
echo ""
echo "2. Or run in background with nohup:"
echo "   cd $INSTALL_DIR"
echo "   nohup ./start_mlflow_server.sh > mlflow.log 2>&1 &"
echo ""
echo "3. Or use systemd (if configured):"
echo "   sudo systemctl start mlflow-server"
echo ""
echo "4. Check if server is running:"
echo "   curl http://localhost:$MLFLOW_PORT/health"
echo ""
echo "5. Access from your local machine (SSH tunnel):"
echo "   ssh -L 5001:localhost:$MLFLOW_PORT \$USER@\$(curl -s http://169.254.169.254/latest/meta-data/public-hostname)"
echo "   Then open: http://localhost:5001"
echo ""
echo "6. Test with the test script:"
echo "   On your local machine:"
echo "   export MLFLOW_TRACKING_USERNAME='$MLFLOW_USERNAME'"
echo "   export MLFLOW_TRACKING_PASSWORD='$MLFLOW_PASSWORD'"
echo "   python scripts/test_remote_mlflow.py"
echo ""
echo "Files created:"
echo "  - $STARTUP_SCRIPT (server startup script)"
echo "  - $AUTH_CONFIG_FILE (authentication config)"
echo "  - $INSTALL_DIR/venv/ (Python virtual environment)"
if [ -f "/etc/systemd/system/mlflow-server.service" ]; then
    echo "  - /etc/systemd/system/mlflow-server.service (systemd service)"
fi
echo ""

print_header "Setup Complete!"

