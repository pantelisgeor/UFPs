import argparse
from src.argparse_to_json.converter import Converter

# From https://github.com/childsish/argparse-to-json

def convert_parser_to_json(parser: argparse.ArgumentParser) -> dict:
    return Converter().convert(parser)
