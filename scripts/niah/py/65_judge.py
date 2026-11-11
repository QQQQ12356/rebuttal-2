"""Judge saved niah predictions."""
from niah_bootstrap import require_linux
from seco.judge import main


if __name__ == "__main__":
    require_linux()
    main("niah")
