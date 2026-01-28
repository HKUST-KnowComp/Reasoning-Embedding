import os
import sys

import torch


def reformat_sd(sd_dir):
    bin_path = os.path.join(sd_dir, "pytorch_model.bin")

    sd = torch.load(bin_path)

    # Check if already reformatted by checking if first key has model. prefix
    if not list(sd.keys())[0].startswith("model."):
        print("SD seems already reformatted: ", sd.keys())
        return

    # Remove model i.e. model.h.1 -> h.1
    sd = {k[6:]: v for k, v in sd.items()}

    torch.save(sd, bin_path)

    print(f"Reformatted SD saved to {bin_path}")


if __name__ == "__main__":
    """
    python3 models/reformat_statedict.py /data/wychanbu/re_data/hotpotqa-instruction-no-negatives/
    """
    reformat_sd(sys.argv[1])
