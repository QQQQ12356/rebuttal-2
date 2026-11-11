"""Train the selected method on RULER-style QA data."""
from ruler_bootstrap import require_linux
from seco.experiment import training_config
from seco.train_cli import main


if __name__ == "__main__":
    require_linux()
    main(training_config("ruler_qa"), expected_task="ruler_qa")
