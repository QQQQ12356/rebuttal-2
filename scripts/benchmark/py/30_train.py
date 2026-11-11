"""Five-dataset training entry; the shared trainer validates the QA data task."""
from _bootstrap import require_linux
from seco.experiment import training_config
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main(default_config=training_config("benchmark"))
