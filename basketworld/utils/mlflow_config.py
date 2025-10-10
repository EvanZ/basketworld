"""
MLflow Configuration Module

This module provides centralized configuration for MLflow tracking and artifact storage.
Supports both local storage and remote S3 storage based on environment variables.

Environment Variables (in order of precedence):
    1. MLFLOW_AWS_ACCESS_KEY_ID: Project-specific AWS access key (takes precedence)
    2. AWS_ACCESS_KEY_ID: Global AWS access key (fallback)

    1. MLFLOW_AWS_SECRET_ACCESS_KEY: Project-specific AWS secret key (takes precedence)
    2. AWS_SECRET_ACCESS_KEY: Global AWS secret key (fallback)

    1. MLFLOW_AWS_DEFAULT_REGION: Project-specific AWS region (takes precedence)
    2. AWS_DEFAULT_REGION: Global AWS region (fallback, default: us-east-1)

    MLFLOW_TRACKING_URI: The tracking URI (default: http://localhost:5000)
    MLFLOW_S3_ENDPOINT_URL: S3 endpoint URL (optional, for custom S3 endpoints)
    MLFLOW_ARTIFACT_ROOT: S3 artifact root URI (e.g., s3://my-bucket/mlflow-artifacts)

.env File Support:
    If a .env file exists in the project root, it will be automatically loaded.
    This allows project-specific credentials without conflicting with global AWS config.

Usage:
    from basketworld.utils.mlflow_config import get_mlflow_config

    config = get_mlflow_config()
    mlflow.set_tracking_uri(config.tracking_uri)
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_env_file(env_path: Optional[Path] = None) -> None:
    """
    Load environment variables from .env file if it exists.

    Args:
        env_path: Path to .env file. If None, looks for .env in project root.
    """
    if env_path is None:
        # Try to find project root (where .env would be)
        current_file = Path(__file__).resolve()
        project_root = (
            current_file.parent.parent.parent
        )  # basketworld/utils/mlflow_config.py -> project root
        env_path = project_root / ".env"

    if not env_path.exists() or not env_path.is_file():
        return

    # Read and parse .env file
    try:
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue

                # Parse KEY=VALUE format
                if "=" in line:
                    key, value = line.split("=", 1)
                    key = key.strip()
                    value = value.strip()
                    # Remove quotes if present
                    if value.startswith('"') and value.endswith('"'):
                        value = value[1:-1]
                    elif value.startswith("'") and value.endswith("'"):
                        value = value[1:-1]

                    # Only set if not already set (don't override existing env vars)
                    if key not in os.environ:
                        os.environ[key] = value
    except (IOError, OSError):
        # Silently skip if we can't read the file
        pass


def _get_aws_credential(
    mlflow_var: str, aws_var: str, default: Optional[str] = None
) -> Optional[str]:
    """
    Get AWS credential with precedence: MLFLOW_* prefix > AWS_* standard > default.

    Args:
        mlflow_var: MLflow-prefixed variable name (e.g., "MLFLOW_AWS_ACCESS_KEY_ID")
        aws_var: Standard AWS variable name (e.g., "AWS_ACCESS_KEY_ID")
        default: Default value if neither is set

    Returns:
        The credential value or None
    """
    return os.environ.get(mlflow_var) or os.environ.get(aws_var) or default


@dataclass
class MLflowConfig:
    """Configuration for MLflow tracking and artifact storage."""

    tracking_uri: str
    artifact_root: Optional[str] = None
    s3_endpoint_url: Optional[str] = None
    use_s3: bool = False
    aws_access_key_id: Optional[str] = None
    aws_secret_access_key: Optional[str] = None
    aws_region: Optional[str] = None

    def __str__(self) -> str:
        """Return a string representation of the configuration."""
        lines = [
            "MLflow Configuration:",
            f"  Tracking URI: {self.tracking_uri}",
            f"  Storage Type: {'S3 (Remote)' if self.use_s3 else 'Local'}",
        ]
        if self.artifact_root:
            lines.append(f"  Artifact Root: {self.artifact_root}")
        if self.s3_endpoint_url:
            lines.append(f"  S3 Endpoint: {self.s3_endpoint_url}")
        if self.use_s3:
            # Show credential source (project-specific vs global)
            cred_source = (
                "project-specific (MLFLOW_AWS_*)"
                if self._is_using_mlflow_prefix()
                else "global (AWS_*)"
            )
            lines.append(f"  AWS Credentials: {cred_source}")
            if self.aws_region:
                lines.append(f"  AWS Region: {self.aws_region}")
        return "\n".join(lines)

    def _is_using_mlflow_prefix(self) -> bool:
        """Check if using MLFLOW_AWS_* prefixed credentials."""
        return os.environ.get("MLFLOW_AWS_ACCESS_KEY_ID") is not None

    def set_boto3_env(self) -> None:
        """
        Set boto3-specific environment variables for this configuration.
        This ensures boto3 uses the project-specific credentials.
        """
        if not self.use_s3:
            return

        # If using MLFLOW_AWS_* credentials, set them as standard AWS_* for boto3
        if self.aws_access_key_id:
            os.environ["AWS_ACCESS_KEY_ID"] = self.aws_access_key_id
        if self.aws_secret_access_key:
            os.environ["AWS_SECRET_ACCESS_KEY"] = self.aws_secret_access_key
        if self.aws_region:
            os.environ["AWS_DEFAULT_REGION"] = self.aws_region


def get_mlflow_config(load_env: bool = True) -> MLflowConfig:
    """
    Get MLflow configuration from environment variables.

    Args:
        load_env: Whether to automatically load .env file if it exists.

    Returns:
        MLflowConfig: Configuration object with tracking URI and artifact settings.

    Examples:
        Local storage (default):
            No environment variables needed.

        S3 storage (project-specific credentials in .env):
            Create .env file:
                MLFLOW_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts
                MLFLOW_AWS_ACCESS_KEY_ID=your-access-key
                MLFLOW_AWS_SECRET_ACCESS_KEY=your-secret-key
                MLFLOW_AWS_DEFAULT_REGION=us-east-1

        S3 storage (global credentials):
            export MLFLOW_ARTIFACT_ROOT=s3://my-bucket/mlflow-artifacts
            export AWS_ACCESS_KEY_ID=your-access-key
            export AWS_SECRET_ACCESS_KEY=your-secret-key
            export AWS_DEFAULT_REGION=us-east-1
    """
    # Load .env file if requested
    if load_env:
        _load_env_file()

    # Get tracking URI (defaults to local server)
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")

    # Get artifact root (if using S3)
    artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT")

    # Get S3 endpoint URL (optional, for custom S3 endpoints like MinIO)
    s3_endpoint_url = os.environ.get("MLFLOW_S3_ENDPOINT_URL")

    # Determine if we're using S3
    use_s3 = artifact_root is not None and artifact_root.startswith("s3://")

    # Get AWS credentials with precedence: MLFLOW_AWS_* > AWS_*
    aws_access_key_id = _get_aws_credential(
        "MLFLOW_AWS_ACCESS_KEY_ID", "AWS_ACCESS_KEY_ID"
    )
    aws_secret_access_key = _get_aws_credential(
        "MLFLOW_AWS_SECRET_ACCESS_KEY", "AWS_SECRET_ACCESS_KEY"
    )
    aws_region = _get_aws_credential(
        "MLFLOW_AWS_DEFAULT_REGION", "AWS_DEFAULT_REGION", "us-east-1"
    )

    return MLflowConfig(
        tracking_uri=tracking_uri,
        artifact_root=artifact_root,
        s3_endpoint_url=s3_endpoint_url,
        use_s3=use_s3,
        aws_access_key_id=aws_access_key_id,
        aws_secret_access_key=aws_secret_access_key,
        aws_region=aws_region,
    )


def setup_mlflow(verbose: bool = True) -> MLflowConfig:
    """
    Set up MLflow with the configured tracking URI and return the config.

    Args:
        verbose: Whether to print configuration details.

    Returns:
        MLflowConfig: The configuration used to set up MLflow.

    Raises:
        ImportError: If boto3 is not installed when using S3 storage.
        ValueError: If AWS credentials are not set when using S3 storage.
    """
    import mlflow

    config = get_mlflow_config()

    # Validate S3 configuration if using S3
    if config.use_s3:
        try:
            import boto3
        except ImportError:
            raise ImportError(
                "boto3 is required for S3 storage. Install it with: pip install boto3"
            )

        # Check for AWS credentials (either MLFLOW_AWS_* or AWS_*)
        if not config.aws_access_key_id:
            raise ValueError(
                "AWS credentials not found. Set either:\n"
                "  - MLFLOW_AWS_ACCESS_KEY_ID (project-specific), or\n"
                "  - AWS_ACCESS_KEY_ID (global)\n"
                "You can also add them to a .env file in the project root."
            )
        if not config.aws_secret_access_key:
            raise ValueError(
                "AWS secret key not found. Set either:\n"
                "  - MLFLOW_AWS_SECRET_ACCESS_KEY (project-specific), or\n"
                "  - AWS_SECRET_ACCESS_KEY (global)\n"
                "You can also add them to a .env file in the project root."
            )

        # Ensure boto3 uses the configured credentials
        config.set_boto3_env()

    # Set tracking URI
    mlflow.set_tracking_uri(config.tracking_uri)

    if verbose:
        print(config)

    return config
