"""Prepare ICAE reconstruction and continuation data on Linux."""
from _bootstrap import require_linux
from icae.pretraining import prepare


if __name__ == "__main__":
    require_linux()
    prepare()
