from pydantic import BaseModel, Field
from typing import Optional, Literal
from pathlib import Path
import yaml
from .paths import paths

class DatasetConfig(BaseModel):
    raw_path: Path
    preprocessed_path: Path
    name: str
    nrows_preprocess: Optional[int] = None
    text_column_index: int = Field(ge=0, description="Index begins from 0")
    label_column_index: int = Field(ge=0)


class PaddingConfig(BaseModel):
    padding: Literal["pre", "post"] = "post"
    truncating: Literal["pre", "post"] = "post"
    maxlen: int = 75


class FeaturizeConfig(BaseModel):
    method: Literal["padding"] = "padding"
    padding: PaddingConfig


class MLFlowConfig(BaseModel):
    perform: bool = True
    experiment_name: str
    tracking_uri: str


class TrainConfig(BaseModel):
    perform: bool = True
    nrows_train: Optional[int] = None
    epochs: int = Field(gt=0)
    batch_size: int = Field(gt=0)
    test_size: float = Field(gt=0, lt=1)


class ModelConfig(BaseModel):
    architecture: Literal["rnn", "lstm", "gru"]
    embedding_dimension: int = Field(gt=0)


class RegistryConfig(BaseModel):
    debug: bool = False


class Config(BaseModel):
    dataset: DatasetConfig
    featurize: FeaturizeConfig
    mlflow: MLFlowConfig
    train: TrainConfig
    model: ModelConfig
    registry: RegistryConfig

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        """Load config from YAML file"""
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        return cls(**data)
    
    def to_yaml(self, path: str | Path) -> None:
        """Save config to YAML file"""
        with open(path, 'w') as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


# Usage example
if __name__ == "__main__":
    # Load config
    config = Config.load(paths.USER_CONFIG)

    