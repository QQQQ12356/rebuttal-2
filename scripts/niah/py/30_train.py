"""Independent NIAH training entry; never mixes the five QA/summarization datasets."""
from niah_bootstrap import require_linux
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main(default_config="configs/niah/train.json", expected_task="niah")
