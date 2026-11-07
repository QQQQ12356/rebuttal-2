"""Export a self-contained SeCo encoder/decoder pair."""
from _bootstrap import require_linux
from seco.merge_cli import main


if __name__ == "__main__":
    require_linux()
    main()
