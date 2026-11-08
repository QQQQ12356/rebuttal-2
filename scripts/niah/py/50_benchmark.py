"""Run the Linux NIAH benchmark on the held-out grid by default."""
from niah_bootstrap import require_linux
from seco.niah.benchmark import main


if __name__ == "__main__":
    require_linux()
    main()
