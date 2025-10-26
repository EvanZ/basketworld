#!/usr/bin/env python3
"""
Test Remote MLflow Server Script

This script tests connectivity and functionality with a remote MLflow server.
It creates a simple sklearn model, logs parameters, metrics, and artifacts.

Usage:
    # Interactive (prompts for credentials)
    python scripts/test_remote_mlflow.py
    
    # With environment variables
    export MLFLOW_TRACKING_USERNAME=your_username
    export MLFLOW_TRACKING_PASSWORD=your_password
    python scripts/test_remote_mlflow.py
    
    # Custom tracking URI
    python scripts/test_remote_mlflow.py --tracking-uri http://localhost:5001
"""

import argparse
import getpass
import os
import sys
import time
from datetime import datetime

import mlflow
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


def setup_credentials():
    """Set up MLflow tracking credentials (username/password)."""
    username = os.environ.get("MLFLOW_TRACKING_USERNAME")
    password = os.environ.get("MLFLOW_TRACKING_PASSWORD")
    
    if not username:
        print("\n" + "=" * 60)
        print("MLflow Server Authentication")
        print("=" * 60)
        username = input("Username: ").strip()
        if not username:
            print("❌ Username is required")
            return False
    
    if not password:
        password = getpass.getpass("Password: ")
        if not password:
            print("❌ Password is required")
            return False
    
    # Set credentials as environment variables for MLflow client
    os.environ["MLFLOW_TRACKING_USERNAME"] = username
    os.environ["MLFLOW_TRACKING_PASSWORD"] = password
    
    return True


def setup_aws_credentials():
    """Set up AWS credentials using the basketworld profile."""
    print("\n" + "=" * 60)
    print("Step 1: Setting up AWS credentials")
    print("=" * 60)
    
    # Check if AWS credentials file exists
    aws_creds_path = os.path.expanduser("~/.aws/credentials")
    if not os.path.exists(aws_creds_path):
        print("⚠️  Warning: ~/.aws/credentials not found")
        print("   S3 artifact access may not work")
        print()
        return False
    
    # Set AWS profile to basketworld (same as local setup)
    os.environ["AWS_PROFILE"] = "basketworld"
    print("✓ AWS_PROFILE set to 'basketworld'")
    
    # Also explicitly set AWS credentials from the profile
    # This is more reliable than relying on AWS_PROFILE alone
    try:
        import configparser
        config = configparser.ConfigParser()
        config.read(aws_creds_path)
        
        if 'basketworld' in config:
            access_key = config['basketworld'].get('aws_access_key_id')
            secret_key = config['basketworld'].get('aws_secret_access_key')
            
            if access_key and secret_key:
                os.environ["AWS_ACCESS_KEY_ID"] = access_key
                os.environ["AWS_SECRET_ACCESS_KEY"] = secret_key
                print("✓ AWS credentials loaded from profile")
            else:
                print("⚠️  Warning: Credentials incomplete in basketworld profile")
                return False
        else:
            print("⚠️  Warning: 'basketworld' profile not found in credentials file")
            return False
        
        # Also check for region in config file
        aws_config_path = os.path.expanduser("~/.aws/config")
        if os.path.exists(aws_config_path):
            config.read(aws_config_path)
            profile_key = 'profile basketworld'
            if profile_key in config:
                region = config[profile_key].get('region', 'us-west-1')
                os.environ["AWS_DEFAULT_REGION"] = region
                print(f"✓ AWS region set to '{region}'")
        
    except Exception as e:
        print(f"⚠️  Warning: Failed to read AWS credentials: {e}")
        return False
    
    print(f"✓ Using credentials from {aws_creds_path}")
    print()
    return True


def test_server_connectivity(tracking_uri):
    """Test basic connectivity to MLflow server."""
    print("=" * 60)
    print("Step 2: Testing server connectivity")
    print("=" * 60)
    
    try:
        mlflow.set_tracking_uri(tracking_uri)
        print(f"✓ Tracking URI set to: {tracking_uri}")
        
        # Try to list experiments
        experiments = mlflow.search_experiments()
        print(f"✓ Successfully connected to MLflow server")
        print(f"✓ Found {len(experiments)} experiment(s)")
        
        # Show some experiment info
        if experiments:
            print("\nExisting experiments:")
            for exp in experiments[:5]:  # Show first 5
                print(f"  - {exp.name} (ID: {exp.experiment_id})")
            if len(experiments) > 5:
                print(f"  ... and {len(experiments) - 5} more")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Failed to connect to MLflow server: {e}")
        print()
        print("Troubleshooting:")
        print("  1. Ensure SSH tunnel is active: ssh -L 5001:localhost:5000 user@ec2-host")
        print("  2. Check username/password are correct")
        print("  3. Verify MLflow server is running on EC2")
        print()
        return False


def create_and_train_model():
    """Create a simple sklearn model and train it."""
    print("=" * 60)
    print("Step 3: Creating and training sklearn model")
    print("=" * 60)
    
    # Generate synthetic dataset
    print("Generating synthetic classification dataset...")
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        random_state=42,
    )
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"✓ Dataset created: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
    
    # Train model
    print("Training logistic regression model...")
    model = LogisticRegression(
        C=1.0,
        max_iter=1000,
        solver="lbfgs",
        random_state=42,
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    print(f"✓ Model trained in {training_time:.2f} seconds")
    
    # Evaluate
    print("Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    metrics = {
        "train_accuracy": accuracy_score(y_train, y_pred_train),
        "test_accuracy": accuracy_score(y_test, y_pred_test),
        "train_f1": f1_score(y_train, y_pred_train),
        "test_f1": f1_score(y_test, y_pred_test),
        "train_precision": precision_score(y_train, y_pred_train),
        "test_precision": precision_score(y_test, y_pred_test),
        "train_recall": recall_score(y_train, y_pred_train),
        "test_recall": recall_score(y_test, y_pred_test),
        "training_time": training_time,
    }
    
    print(f"✓ Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"✓ Test F1 score: {metrics['test_f1']:.4f}")
    print()
    
    return model, metrics, (X_test, y_test)


def log_to_mlflow(model, metrics, test_data):
    """Log model, parameters, and metrics to MLflow."""
    print("=" * 60)
    print("Step 4: Logging to MLflow")
    print("=" * 60)
    
    experiment_name = "remote_mlflow_test"
    
    try:
        # Set or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"✓ Created new experiment: '{experiment_name}' (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            print(f"✓ Using existing experiment: '{experiment_name}' (ID: {experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        
        # Start run
        run_name = f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"Starting run: {run_name}")
        print()
        
        with mlflow.start_run(run_name=run_name) as run:
            run_id = run.info.run_id
            artifact_uri = run.info.artifact_uri
            
            print(f"Run ID: {run_id}")
            print(f"Artifact URI: {artifact_uri}")
            print()
            
            # Log parameters
            print("Logging parameters...")
            params = {
                "model_type": "LogisticRegression",
                "C": 1.0,
                "max_iter": 1000,
                "solver": "lbfgs",
                "random_state": 42,
                "n_features": 20,
                "n_samples": 1000,
            }
            mlflow.log_params(params)
            print(f"✓ Logged {len(params)} parameters")
            
            # Log metrics
            print("Logging metrics...")
            mlflow.log_metrics(metrics)
            print(f"✓ Logged {len(metrics)} metrics")
            
            # Log model
            print("Logging model...")
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=None,  # Don't register for test
            )
            print("✓ Model logged successfully")
            
            # Log additional artifacts
            print("Logging additional artifacts...")
            
            # Create a simple text artifact
            artifact_text = f"""MLflow Remote Server Test
================================
Timestamp: {datetime.now().isoformat()}
Run ID: {run_id}
Model Type: Logistic Regression
Test Accuracy: {metrics['test_accuracy']:.4f}
Test F1 Score: {metrics['test_f1']:.4f}

This is a test run to verify the remote MLflow server configuration.
"""
            
            # Save text artifact
            with open("/tmp/test_info.txt", "w") as f:
                f.write(artifact_text)
            mlflow.log_artifact("/tmp/test_info.txt", artifact_path="info")
            print("✓ Logged text artifact")
            
            # Create and log a simple numpy array artifact
            X_test, y_test = test_data
            predictions = model.predict(X_test)
            np.save("/tmp/test_predictions.npy", predictions)
            mlflow.log_artifact("/tmp/test_predictions.npy", artifact_path="predictions")
            print("✓ Logged predictions artifact")
            
            # Log tags
            print("Logging tags...")
            mlflow.set_tags({
                "purpose": "server_test",
                "test_type": "connectivity_and_functionality",
                "environment": "remote_ec2",
            })
            print("✓ Logged tags")
            
            print()
            print(f"✓ Run completed successfully!")
            print(f"  View in MLflow UI: {mlflow.get_tracking_uri()}/experiments/{experiment_id}/runs/{run_id}")
            print()
            
            return run_id
            
    except Exception as e:
        print(f"❌ Failed to log to MLflow: {e}")
        print()
        import traceback
        traceback.print_exc()
        return None


def test_artifact_retrieval(run_id):
    """Test retrieving artifacts from the logged run."""
    print("=" * 60)
    print("Step 5: Testing artifact retrieval")
    print("=" * 60)
    
    try:
        # Get run info
        run = mlflow.get_run(run_id)
        print(f"✓ Retrieved run info for: {run_id}")
        
        # List artifacts
        client = mlflow.tracking.MlflowClient()
        artifacts = client.list_artifacts(run_id)
        print(f"✓ Found {len(artifacts)} artifact(s):")
        for artifact in artifacts:
            print(f"  - {artifact.path} ({artifact.file_size} bytes)")
        
        # Try to download an artifact
        print("\nAttempting to download artifact...")
        artifact_path = client.download_artifacts(run_id, "info/test_info.txt")
        with open(artifact_path, "r") as f:
            content = f.read()
        print(f"✓ Successfully downloaded artifact")
        print(f"  Content preview: {content[:100]}...")
        
        print()
        return True
        
    except Exception as e:
        print(f"❌ Failed to retrieve artifacts: {e}")
        print()
        import traceback
        traceback.print_exc()
        return False


def cleanup_temp_files():
    """Clean up temporary test files."""
    temp_files = ["/tmp/test_info.txt", "/tmp/test_predictions.npy"]
    for filepath in temp_files:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except Exception:
            pass


def main():
    """Main test routine."""
    parser = argparse.ArgumentParser(description="Test remote MLflow server")
    parser.add_argument(
        "--tracking-uri",
        default="http://localhost:5001",
        help="MLflow tracking URI (default: http://localhost:5001)",
    )
    parser.add_argument(
        "--skip-model",
        action="store_true",
        help="Skip model training (only test connectivity)",
    )
    args = parser.parse_args()
    
    print("\n" + "=" * 60)
    print("MLflow Remote Server Test")
    print("=" * 60)
    print(f"Tracking URI: {args.tracking_uri}")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()
    
    # Setup AWS credentials
    aws_ok = setup_aws_credentials()
    if not aws_ok:
        print("⚠️  Continuing without AWS credentials...")
        print()
    
    # Setup MLflow credentials
    if not setup_credentials():
        print("❌ Failed to set up credentials")
        return 1
    
    # Test server connectivity
    if not test_server_connectivity(args.tracking_uri):
        return 1
    
    # If only testing connectivity, stop here
    if args.skip_model:
        print("✅ Connectivity test passed! (Skipped model training)")
        return 0
    
    # Create and train model
    model, metrics, test_data = create_and_train_model()
    
    # Log to MLflow
    run_id = log_to_mlflow(model, metrics, test_data)
    if not run_id:
        cleanup_temp_files()
        return 1
    
    # Test artifact retrieval
    if not test_artifact_retrieval(run_id):
        cleanup_temp_files()
        return 1
    
    # Cleanup
    cleanup_temp_files()
    
    # Final summary
    print("=" * 60)
    print("✅ ALL TESTS PASSED!")
    print("=" * 60)
    print()
    print("Summary:")
    print("  ✓ Server connectivity")
    print("  ✓ Authentication")
    print("  ✓ Model training")
    print("  ✓ Parameter logging")
    print("  ✓ Metric logging")
    print("  ✓ Model logging")
    print("  ✓ Artifact logging")
    print("  ✓ Artifact retrieval")
    print()
    print("Your remote MLflow server is working correctly!")
    print(f"Run ID: {run_id}")
    print()
    
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n❌ Test interrupted by user")
        cleanup_temp_files()
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        cleanup_temp_files()
        sys.exit(1)

