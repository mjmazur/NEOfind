"""Command-line parsing (``neodetect.cli``)."""

import argparse

from neodetect.cli import build_parser


def test_every_command_formats_its_help():
    # argparse %-formats help strings: a bare "%" breaks --help.
    p = build_parser()
    sub = next(a for a in p._actions if isinstance(a, argparse._SubParsersAction))
    for name, parser in sub.choices.items():
        assert parser.format_help(), name


def test_skip_crowded_defaults_to_the_instrument():
    args = build_parser().parse_args(["detect", "somewhere"])
    assert args.skip_crowded is None
    assert build_parser().parse_args(["detect", "x", "--no-skip-crowded"]).skip_crowded is False
