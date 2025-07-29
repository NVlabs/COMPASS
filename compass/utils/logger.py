"""
Flexible logging system that supports both Weights & Biases and TensorBoard.
"""

import os
from typing import Dict, Any, Optional

import wandb
from tensorboardX import SummaryWriter


class Logger:
    """
    A unified logger that can use either Weights & Biases or TensorBoard.
    """

    def __init__(self,
                 log_dir: str,
                 experiment_name: str,
                 backend: str = "tensorboard",
                 project_name: str = "compass",
                 entity: Optional[str] = None):
        """
        Initialize the logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            backend: Either "wandb" or "tensorboard"
            project_name: Project name (for wandb)
            entity: Team/entity name (for wandb)
            config: Configuration dictionary to log
        """
        self.backend = backend.lower()
        self.experiment_name = experiment_name
        self.log_dir = log_dir

        # Create log directory if it doesn't exist
        os.makedirs(self.log_dir, exist_ok=True)

        # Initialize the backend.
        if self.backend == "wandb":
            # Initialize wandb
            self.writer = wandb.init(
                project=project_name,
                entity=entity,
                name=experiment_name,
                dir=log_dir,
            )
        elif self.backend == "tensorboard":
            # Initialize TensorBoard writer
            self.writer = SummaryWriter(os.path.join(self.log_dir, "tensorboard"))
        else:
            raise ValueError(f"Unsupported backend: {backend}. "
                             "Choose either 'wandb' or 'tensorboard'.")

        print(f"Logger initialized with backend: {self.backend}")

    def log_video(self, name: str, video_path: str, fps: int = 4, step: Optional[int] = None):
        """
        Log a video.

        Args:
            name: Name of the video
            video_path: Path to the video file
            fps: Frames per second
            step: Step number (if None, uses 0)
        """
        if step is None:
            step = 0

        if self.backend == "wandb":
            wandb.log(
                {name: wandb.Video(video_path, fps=fps, format="mp4", caption=f"Iteration {step}")})
        else:
            print(f"Video upload via path not supported in TensorBoard. Video '{name}' not logged.")

    def log_dict(self, dict: Dict[str, Any], step: Optional[int] = None):
        """
        Log multiple metrics at once.

        Args:
            dict: Dictionary of scalars to log
            step: Step number (if None, uses 0)
        """
        if self.backend == "wandb":
            wandb.log(dict, step=step)
        else:
            for name, value in dict.items():
                self.writer.add_scalar(name, value, global_step=step)

    def log_artifact(self,
                     artifact_path: str,
                     name: str,
                     type: str,
                     description: Optional[str] = None):
        """
        Log an artifact (file or directory).

        Args:
            artifact_path: Path to the artifact
            name: Name of the artifact
            type: Type of the artifact
            description: Description of the artifact
        """
        if self.backend == "wandb":
            artifact = wandb.Artifact(name=name, type=type, description=description)
            artifact.add_file(artifact_path)
            wandb.log_artifact(artifact)
        else:
            # TensorBoard doesn't support artifacts directly
            print(f"Artifacts not supported in TensorBoard. Artifact '{name}' not logged.")

    def log_config(self, config: Dict[str, Any]):
        """
        Update the run configuration.

        Args:
            config: Configuration dictionary
        """
        if self.backend == "wandb":
            self.writer.config.update(config)
        else:
            # TensorBoard doesn't support updating config
            print("Updating config not supported in TensorBoard.")

    def close(self):
        """Close the logger."""
        if self.backend == "wandb":
            wandb.finish()
        elif self.backend == "tensorboard":
            self.writer.close()
