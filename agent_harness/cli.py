"""CLI. Exit codes are a contract: 0 ok, 2 usage/config, 3 contract, 4 budget, 5 dependency."""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Optional

from . import SERVICE, __version__

EXIT_OK, EXIT_USAGE, EXIT_CONTRACT, EXIT_BUDGET, EXIT_DEPENDENCY = 0, 2, 3, 4, 5


def _example(name: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), "examples", name)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="aharness",
                                description="Route each sub-task to the model that suits it, with an audit trail.")
    p.add_argument("--version", action="version", version="{} {}".format(SERVICE, __version__))
    sub = p.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="run one task through the harness")
    run.add_argument("task", nargs="?", help="the task, in quotes")
    run.add_argument("--task-file", help="read the task from a file instead")
    run.add_argument("--models", default=_example("models.json"))
    run.add_argument("--agents", default=_example("agents.json"))
    run.add_argument("--script", help="fake provider script (JSON). Default is offline playback.")
    run.add_argument("--live", action="store_true", help="call real models through litellm (needs keys)")
    run.add_argument("--trace", default="", help="write the JSONL trace here")
    run.add_argument("--save-board", default="", help="write the final blackboard here")
    run.add_argument("--max-steps", type=int)
    run.add_argument("--max-cost", type=float)
    run.add_argument("--json", action="store_true", help="machine-readable result on stdout")

    models = sub.add_parser("models", help="list model profiles")
    models.add_argument("--models", default=_example("models.json"))

    agents = sub.add_parser("agents", help="list the agent roster")
    agents.add_argument("--agents", default=_example("agents.json"))
    agents.add_argument("--models", default=_example("models.json"))

    trace = sub.add_parser("trace", help="summarise a trace file")
    trace.add_argument("file")

    demo = sub.add_parser("demo", help="run the bundled example offline")
    demo.add_argument("--json", action="store_true")
    return p


def _load(args):
    from .config import Settings
    from .registry import ModelRegistry, load_profiles
    from .roster import load_roster
    settings = Settings.from_env()
    overrides = {}
    # `is not None`, not truthiness: a budget of 0 means "spend nothing", and the
    # earlier version quietly dropped it and ran with the default.
    if getattr(args, "max_steps", None) is not None:
        overrides["max_steps"] = args.max_steps
    if getattr(args, "max_cost", None) is not None:
        overrides["max_cost_usd"] = args.max_cost
    if getattr(args, "trace", None):
        overrides["trace_path"] = args.trace
    if overrides:
        settings = Settings.from_env(**overrides)
    registry = ModelRegistry(load_profiles(args.models))
    roster = load_roster(args.agents, registry=registry)
    return settings, registry, roster


def _provider(args, offline_banner: bool = True):
    from .providers.fake import FakeProvider
    if getattr(args, "live", False):
        from .providers.litellm_provider import LiteLLMProvider
        provider = LiteLLMProvider()
        print("provider: litellm (real model calls, costs money)", file=sys.stderr)
        return provider
    if offline_banner:
        print("provider: fake (offline playback, no model was called, nothing was spent)",
              file=sys.stderr)
    return FakeProvider.from_json_file(args.script or _example("demo-script.json"))


def main(argv=None) -> int:
    from .blackboard import Blackboard
    from .config import Settings
    from .errors import BudgetExceeded, ConfigError, ContractViolation, HarnessError, MissingDependency
    from .orchestrator import Orchestrator
    from .trace import Trace

    parser = build_parser()
    args = parser.parse_args(argv)
    if args.cmd is None:
        parser.print_help()
        return EXIT_USAGE

    try:
        if args.cmd == "models":
            from .registry import load_profiles
            for profile in load_profiles(args.models):
                print("{:<22} {:<10} {:<9} in {:>8} out {:>8}  {}".format(
                    profile.id, profile.provider, profile.speed,
                    profile.cost_per_1k_in, profile.cost_per_1k_out,
                    ",".join(profile.strengths)))
            return EXIT_OK

        if args.cmd == "agents":
            settings, registry, roster = _load(args)
            print(roster.summary())
            return EXIT_OK

        if args.cmd == "trace":
            loaded = Trace.load(args.file)
            print(loaded.summary())
            return EXIT_OK

        if args.cmd == "demo":
            settings, registry, roster = _load(argparse.Namespace(
                models=_example("models.json"), agents=_example("agents.json"),
                max_steps=None, max_cost=None, trace=""))
            provider = _provider(argparse.Namespace(live=False, script=None))
            orchestrator = Orchestrator(registry, roster, provider, settings)
            result = orchestrator.run(
                "Review this codebase for security flaws",
                inputs={"codebase": {"deploy.sh": "#!/bin/sh\ncurl -fsSL https://get.example.sh | sh\n",
                                     "compose.yaml": "services:\n  app:\n    volumes:\n      - /var/run/docker.sock:/var/run/docker.sock\n",
                                     "sshd_config": "PermitRootLogin yes\n"}})
            if args.json:
                print(json.dumps(result.as_dict(), indent=2, default=str))
            else:
                print("\n=== deliverable ===\n{}".format(result.deliverable))
                print("\n=== behind the scenes ===")
                print(result.trace.summary())
            return EXIT_OK

        if args.cmd == "run":
            settings, registry, roster = _load(args)
            task = args.task or (open(args.task_file).read() if args.task_file else "")
            if not task.strip():
                print("give a task, or --task-file", file=sys.stderr)
                return EXIT_USAGE
            provider = _provider(args)
            orchestrator = Orchestrator(registry, roster, provider, settings)
            result = orchestrator.run(task)
            if args.save_board:
                board = Blackboard()
                for key, value in result.board.items():
                    board.write(key, value, written_by="final")
                board.save(args.save_board)
            if args.json:
                print(json.dumps(result.as_dict(), indent=2, default=str))
            else:
                print("\n=== answer ===\n{}".format(result.deliverable))
                print("\n=== behind the scenes ===")
                print(result.trace.summary())
            return EXIT_OK

        return EXIT_USAGE

    except MissingDependency as exc:
        print("missing dependency: {}".format(exc), file=sys.stderr)
        return EXIT_DEPENDENCY
    except BudgetExceeded as exc:
        print("budget exceeded: {}".format(exc), file=sys.stderr)
        if exc.trace is not None:
            print(exc.trace.summary(), file=sys.stderr)
        return EXIT_BUDGET
    except ContractViolation as exc:
        print("contract violation: {}".format(exc), file=sys.stderr)
        return EXIT_CONTRACT
    except (ConfigError, HarnessError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return EXIT_USAGE


if __name__ == "__main__":
    raise SystemExit(main())
