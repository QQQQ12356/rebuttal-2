"""Options for recording SECO group sizes during benchmark generation."""
from argparse import BooleanOptionalAction


def add_group_stats_argument(parser):
    parser.add_argument(
        "--record-group-stats",
        action=BooleanOptionalAction,
        help="Save per-sample group sizes in predictions (enabled by default for SECO)",
    )


def resolve_group_stats(parser, requested, model_config):
    is_seco = model_config.get("method", "seco") == "seco"
    if requested and not is_seco:
        parser.error("--record-group-stats is only supported for SECO")
    return is_seco if requested is None else requested
