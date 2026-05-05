from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.run import build_parser, run_single_stream


def main():
    parser = build_parser(default_model_name="transunet", dual_stream=False)
    parser.set_defaults(input_mode_l4s="rgb", input_mode_bijie="rgb")
    args = parser.parse_args()
    run_single_stream(args)


if __name__ == "__main__":
    main()
