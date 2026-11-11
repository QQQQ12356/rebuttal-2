"""Independent NIAH training entry; never mixes the five QA/summarization datasets."""
from niah_bootstrap import require_linux
from seco.experiment import training_config
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main(default_config=training_config("niah"), expected_task="niah")
