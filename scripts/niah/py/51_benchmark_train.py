"""Run explicit NIAH training-set evaluation on Linux."""
from niah_bootstrap import require_linux
from seco.niah.benchmark import main


if __name__ == "__main__":
    require_linux()
    main(train_only=True)
