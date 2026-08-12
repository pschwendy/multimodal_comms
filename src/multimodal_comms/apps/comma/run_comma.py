#!/usr/bin/env python3
"""Entrypoint that installs the channel-compression middleware, then runs COMMA.

Same CLI as main.py, plus compressor selection via env vars
(COMMA_COMPRESSOR / COMMA_COMPRESSOR_PARAMS) read by channel_adapter.activate().

Usage (headless, under the extracted-Xvfb wrapper):
  run_xvfb.sh 99 python run_comma.py --puzzle_config config/... \
      --model_config config/... --save_folder outputs/...
"""
import argparse
import time

from multimodal_comms.apps.comma import channel_adapter
import multimodal_comms.apps.comma.main as main


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sound", action="store_true", default=False)
    parser.add_argument("--puzzle_config", type=str, default=None)
    parser.add_argument("--model_config", type=str, default=None)
    parser.add_argument("--save_folder", type=str, default="./outputs")
    parser.add_argument("--baseline", type=str, default="none")
    args = parser.parse_args()

    start_time = time.time()
    channel_adapter.activate()
    main.main(args)
