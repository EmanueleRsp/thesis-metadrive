import hydra
from omegaconf import DictConfig

from thesis_rl.runtime.loops.eval_loop import run_evaluation

__all__ = ["main"]


@hydra.main(version_base=None, config_path="../../../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    run_evaluation(cfg)

if __name__ == "__main__":
    main()
