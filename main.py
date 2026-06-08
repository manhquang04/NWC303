"""Top-level entry point."""

from __future__ import annotations

import argparse
import logging
import sys

from evaluation.logger import setup_logging

log = logging.getLogger(__name__)


def _cmd_train(rest: list[str]) -> int:
    from agent.train import main as train_main
    sys.argv = ["train", *rest]
    return train_main()


def _cmd_evaluate(rest: list[str]) -> int:
    from evaluation.metrics import main as eval_main
    sys.argv = ["evaluate", *rest]
    return eval_main()


def _cmd_experiment(rest: list[str]) -> int:
    from evaluation.metrics import main as eval_main
    sys.argv = ["experiment", "--scenario", "all", "--agent", "all", *rest]
    return eval_main()


def _cmd_realtest(rest: list[str]) -> int:
    from evaluation.realtest import main as realtest_main
    sys.argv = ["realtest", *rest]
    return realtest_main()


def _cmd_topo(_rest: list[str]) -> int:
    from env.topology import main as topo_main
    topo_main()
    return 0


def _cmd_plot(_rest: list[str]) -> int:
    from evaluation.visualizer import plot_reward_curve
    plot_reward_curve()
    return 0


def _cmd_web(rest: list[str]) -> int:
    import argparse
    import uvicorn
    p = argparse.ArgumentParser(description="Start web dashboard.")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--ryu-url", default=None)
    p.add_argument("--dpids", default="1,2,3")
    p.add_argument("--sniff-iface", default="any")
    p.add_argument("--no-sdn", action="store_true",
                   help="Run without Mininet connection (demo mode).")
    args, _ = p.parse_known_args(rest)

    from web.app import app, set_bridge
    from web.bridge import WebBridge

    bridge = WebBridge()
    set_bridge(bridge)

    if not args.no_sdn:
        try:
            from detection.flow_collector import FlowCollector
            from isolation.isolator import Isolator

            dpids = [int(x.strip(), 0) for x in args.dpids.split(",") if x.strip()]
            collector_kwargs = {"dpids": dpids}
            if args.ryu_url:
                collector_kwargs["base_url"] = args.ryu_url
            collector = FlowCollector(**collector_kwargs)
            collector.start()
            isolator = Isolator(base_url=args.ryu_url) if args.ryu_url else Isolator()
            bridge.flow_collector = collector
            bridge.isolator = isolator
            bridge.start_sniffer(iface=args.sniff_iface)
            log.info("SDN modules connected (FlowCollector + Isolator + Sniffer), dpids=%s.", dpids)
        except Exception as exc:
            log.warning("Cannot connect SDN modules: %s — running in demo mode.", exc)

    log.info("Starting web dashboard at http://%s:%d", args.host, args.port)
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


COMMANDS = {
    "train": _cmd_train,
    "evaluate": _cmd_evaluate,
    "experiment": _cmd_experiment,
    "realtest": _cmd_realtest,
    "topo": _cmd_topo,
    "plot": _cmd_plot,
    "web": _cmd_web,
}


def main() -> int:
    setup_logging()
    parser = argparse.ArgumentParser(description="SDN-DRL-IDS entry point.")
    parser.add_argument("command", choices=list(COMMANDS.keys()))
    args, rest = parser.parse_known_args()
    return COMMANDS[args.command](rest)


if __name__ == "__main__":
    sys.exit(main())
