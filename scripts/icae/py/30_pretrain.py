"""Train ICAE on reconstruction and continuation targets."""
from _bootstrap import require_linux
from icae.pretraining import TASK
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main("configs/icae/pretrain.json", expected_task=TASK)
