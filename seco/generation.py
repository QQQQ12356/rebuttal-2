"""Shared decoding options for SeCo evaluations."""

import math


def add_generation_arguments(parser):
    parser.add_argument("--decoding", choices=("greedy", "sample", "beam"), default="greedy")
    parser.add_argument("--temperature", type=float, help="Sampling temperature (default: 1.0)")
    parser.add_argument("--top-p", type=float, help="Sampling nucleus cutoff (default: 1.0)")
    parser.add_argument("--top-k", type=int, help="Sampling vocabulary cutoff (default: 0, disabled)")
    parser.add_argument("--num-beams", type=int, help="Beam width (beam mode only, default: 4)")
    parser.add_argument("--min-new-tokens", type=int, default=0, help="Suppress EOS until this many tokens")
    parser.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty (default: 1.0)")


def generation_options(parser, args, max_new_tokens):
    if not 0 <= args.min_new_tokens <= max_new_tokens or max_new_tokens < 1:
        parser.error("Require 0 <= --min-new-tokens <= the maximum generation length, with a positive maximum")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be finite and positive")
    if args.num_beams is not None and (args.decoding != "beam" or args.num_beams < 2):
        parser.error("--num-beams requires --decoding beam and a value of at least 2")
    if args.decoding != "sample" and any(
        value is not None for value in (args.temperature, args.top_p, args.top_k)
    ):
        parser.error("--temperature, --top-p and --top-k require --decoding sample")
    temperature = 1.0 if args.temperature is None else args.temperature
    top_p = 1.0 if args.top_p is None else args.top_p
    top_k = 0 if args.top_k is None else args.top_k
    if not math.isfinite(temperature) or temperature <= 0 or not 0 < top_p <= 1 or top_k < 0:
        parser.error("Require finite temperature > 0, 0 < top-p <= 1, and top-k >= 0")

    return {
        "max_new_tokens": max_new_tokens,
        "min_new_tokens": args.min_new_tokens,
        "do_sample": args.decoding == "sample",
        "num_beams": (args.num_beams or 4) if args.decoding == "beam" else 1,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k if args.decoding == "sample" else 50,
        "repetition_penalty": args.repetition_penalty,
        "num_return_sequences": 1,
        "return_dict_in_generate": False,
        "output_scores": False,
    }
