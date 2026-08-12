#!/usr/bin/env python3
"""End-to-end demo of the private-keyed secure broadcast pipeline on a real
trained checkpoint (not synthetic latents): N different messages -> ONE
packet -> each receiver decodes only its own message via its own private
secret; a receiver given someone else's secret gets garbage, not the wrong
person's actual message.

Uses data/superpose_pretrain_s2/final (Qwen3-4B AE, fine-tuned specifically
for OrthogonalKeyring "qr" rotation crosstalk, K=4 latents) -- see
reports/multiplex_load_curve_20260719.md for this checkpoint's public-mode
numbers; this script reuses the exact same bind/unbind math (only where the
per-slot key comes from differs), so reconstruction quality is expected to
match that report's load-N figures.

Example:
  python experiments/crypt_ae/programs/demo_crypto_broadcast.py --device cuda:0
"""

import argparse
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))

from multimodal_comms.methods.superposition.latent import (  # noqa: E402
    LatentCodec,
    SecureBroadcastCodec,
    SecureReceiverCodec,
    mint_receiver_secrets,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", type=str,
                     default="data/superpose_pretrain_s2/final")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    messages = {
        0: "The quarterly budget review is scheduled for next Tuesday morning.",
        1: "Remember to water the tomato plants before the weekend heat wave.",
        2: "The satellite launch window opens at 0300 UTC on the fourteenth.",
        3: "Grandma's recipe calls for two cups of flour and a pinch of salt.",
    }

    codec = LatentCodec(model_path=args.model_path, device=args.device)
    secrets = mint_receiver_secrets(len(messages))
    print(f"Minted {len(secrets)} independent private receiver secrets "
          f"(63-bit each, unrelated to slot index).\n")

    sender = SecureBroadcastCodec(codec, secrets_by_slot=secrets, key_mode="qr")
    packet = sender.encode_packet(messages)
    print(f"Sent ONE packet ({len(packet)} chars, {len(messages)} messages "
          f"entangled into it) to all {len(messages)} receivers.\n")

    print("=== Each receiver decodes with its OWN secret ===")
    for slot, secret in secrets.items():
        receiver = SecureReceiverCodec(codec, my_slot=slot, my_secret=secret,
                                        key_mode="qr")
        decoded = receiver.decode(packet)
        print(f"[receiver {slot}] true:    {messages[slot]!r}")
        print(f"[receiver {slot}] decoded: {decoded!r}\n")

    print("=== Receiver 0 tries to read receiver 1's message using "
          "receiver 0's OWN (wrong) secret for slot 1 ===")
    eavesdropper = SecureReceiverCodec(codec, my_slot=1, my_secret=secrets[0],
                                        key_mode="qr")
    forged_decode = eavesdropper.decode(packet)
    print(f"[eavesdropper] receiver 1's true message: {messages[1]!r}")
    print(f"[eavesdropper] what slot-0's secret decodes to: {forged_decode!r}")
    print(f"[eavesdropper] matches receiver 1's message: "
          f"{forged_decode == messages[1]}")


if __name__ == "__main__":
    main()
