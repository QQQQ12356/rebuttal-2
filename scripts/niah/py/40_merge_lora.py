"""Export the NIAH SeCo checkpoint with the shared validated LoRA merge routine."""
from niah_bootstrap import require_linux
from seco.merge_cli import main


if __name__ == "__main__":
    require_linux()
    main()
