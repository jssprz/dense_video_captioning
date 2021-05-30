import os
import argparse
import json

from configuration_file import ConfigurationFile
from windows_size_model import WindowsSizeModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a captioning model for a specific dataset."
    )
    parser.add_argument(
        "-m", "--mode", type=str, choices=["train", "predict"], default="train", help=""
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="the path to the config file with all params",
    )

    args = parser.parse_args()

    config = ConfigurationFile(args.config, "WINDOW-SIZE-MODEL")

    model = WindowsSizeModel(config)

    if args.mode == "train":
        print(model.train())
        model.save_model("models/ws.cluster")
    elif args.mode == "predict":
        with open(os.path.join(config.data_dir, "captions/val_1.json")) as f:
            data = json.load(f)

        model.load_model("models/ws.cluster")
        print(model.get_windows_sizes(data))
