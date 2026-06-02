import hydra
from omegaconf import DictConfig

from thesis_rl.runtime.loops.train_loop import _missing_curriculum_metrics, run_training

__all__ = ["main", "_missing_curriculum_metrics"]


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_training(cfg)

if __name__ == "__main__":
    main()
