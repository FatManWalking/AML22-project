from pytorch_lightning.callbacks import ModelCheckpoint
from typing import Dict, Optional, Any


def get_model_checkpoint(args: Dict[str, Any]) -> Optional[ModelCheckpoint]:
    if not args["output_path"]:
        return None
    # Note: It is important that each rank behaves the same.
    # All of the ranks, or none of them should return ModelCheckpoint
    # Otherwise, there will be deadlock for distributed training
    return ModelCheckpoint(
        monitor="train_loss",
        dirpath=args["output_path"],
        save_last=True,
    )
