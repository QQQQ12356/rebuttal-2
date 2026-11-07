"""Export a RULER QA SeCo checkpoint using the shared LoRA merge routine."""
from ruler_bootstrap import require_linux
from seco.merge_cli import main


if __name__ == "__main__":
    require_linux()
    main()
