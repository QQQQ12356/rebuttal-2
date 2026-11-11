"""Judge saved ruler_qa predictions."""
from ruler_bootstrap import require_linux
from seco.judge import main


if __name__ == "__main__":
    require_linux()
    main("ruler_qa")
