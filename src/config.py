from dataclasses import dataclass, field
from omegaconf import DictConfig, OmegaConf

@dataclass
class CVConfig:
    n_folds: int = 5

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "data"
    submission_path: str = "submissions/${model}_submission.csv"
    model: str = "random_forest"
    cv: CVConfig = field(default_factory=CVConfig)
    params: dict = field(default_factory=dict)

def load_config(default_path: str = "configs/config.yaml") -> DictConfig:
    cli = OmegaConf.from_cli()
    path = cli.pop("config", default_path)
    schema = OmegaConf.structured(Config)
    return OmegaConf.merge(schema, OmegaConf.load(path), cli)
