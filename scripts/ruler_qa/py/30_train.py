"""Train SeCo adapters on RULER-style QA data."""
from ruler_bootstrap import require_linux
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main("configs/ruler_qa/train.json", expected_task="ruler_qa")
