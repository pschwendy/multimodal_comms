from __future__ import annotations

import argparse
import json
from collections.abc import Sequence

from multimodal_comms.core import Message, MethodContext
from multimodal_comms.evaluation import load_experiment, run_experiment
from multimodal_comms.registry import create_method, get_method_spec, list_methods


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(
        prog="multimodal-comms", description="Benchmark-neutral communication methods"
    )
    commands = result.add_subparsers(dest="command", required=True)
    commands.add_parser("methods", help="list registered methods")
    roundtrip = commands.add_parser("roundtrip", help="round-trip one text message")
    roundtrip.add_argument("method")
    roundtrip.add_argument("text")
    roundtrip.add_argument("--seed", type=int, default=0)
    run = commands.add_parser("run", help="run a YAML/JSON experiment specification")
    run.add_argument("spec")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "methods":
        for spec in list_methods():
            extras = f" dependencies={','.join(spec.dependencies)}" if spec.dependencies else ""
            print(f"{spec.id:22} {spec.kind:13} {','.join(spec.representation)}{extras}")
        return 0
    if args.command == "roundtrip":
        spec = get_method_spec(args.method)
        if spec.kind != "communication":
            raise SystemExit(f"{args.method!r} is a {spec.kind}, not a communication method")
        method = create_method(args.method)
        context = MethodContext(seed=args.seed)
        transmission = method.encode([Message("cli", None, args.text)], context)
        decoded = method.decode(transmission, context)
        print(decoded[0].content if decoded else "")
        return 0
    print(json.dumps(run_experiment(load_experiment(args.spec)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
