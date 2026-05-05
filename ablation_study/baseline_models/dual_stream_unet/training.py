from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parents[1]))
from common.run import build_parser, run_dual_stream


def main():
    parser = build_parser(default_model_name="dual_stream_unet", dual_stream=True)
    args = parser.parse_args()
    run_dual_stream(args)


if __name__ == "__main__":
    main()
