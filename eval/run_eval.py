"""
eval/run_eval.py — Measure planning accuracy and execution success.

For each benchmark instruction the agent runs end to end; we compare the set
of tools it called against the tools a correct plan requires.

    python eval/run_eval.py                    # offline planner (baseline)
    python eval/run_eval.py --planner claude   # needs ANTHROPIC_API_KEY

Metrics per task and overall:
  tool_recall     required tools that were called / required tools
  tool_precision  called tools that were required / distinct tools called
  exec_success    every tool call succeeded and a non-empty answer came back
  exact_cover     all required tools called (recall == 1) and exec_success
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

DEFAULT_BBOX = [77.50, 12.90, 77.56, 12.96]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--planner", choices=["offline", "claude"], default="offline")
    p.add_argument("--benchmark", default=os.path.join(ROOT, "eval", "benchmark.jsonl"))
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX)
    p.add_argument("--out", default=None, help="Write per-task results as JSON")
    args = p.parse_args()

    from agent import GeoVLAAgent
    import config

    if args.planner == "claude" and not config.llm_enabled():
        raise SystemExit("ANTHROPIC_API_KEY is not set")

    tasks = [json.loads(line) for line in open(args.benchmark) if line.strip()]
    rows = []
    for task in tasks:
        start = time.perf_counter()
        agent = GeoVLAAgent(bbox=args.bbox, use_llm=args.planner == "claude")
        try:
            result = agent.run(task["instruction"])
            answer, trace = result["answer"], result["trace"]
        except Exception as exc:
            answer, trace = f"CRASH: {exc}", agent.trace
        calls = [t for t in trace if t["type"] == "tool_call"]
        called, required = {c["tool"] for c in calls}, set(task["required_tools"])
        recall = len(called & required) / len(required)
        precision = len(called & required) / len(called) if called else 0.0
        success = bool(answer.strip()) and not answer.startswith(("CRASH", "Stopped")) \
            and not any(c["is_error"] for c in calls)
        rows.append({"id": task["id"], "tool_recall": recall, "tool_precision": precision,
                     "exec_success": success, "exact_cover": success and recall == 1.0,
                     "n_calls": len(calls), "seconds": round(time.perf_counter() - start, 2),
                     "missing": sorted(required - called), "extra": sorted(called - required),
                     "answer": answer})
        r = rows[-1]
        print(f"{r['id']}  recall={r['tool_recall']:.2f}  precision={r['tool_precision']:.2f}  "
              f"ok={r['exec_success']}  calls={r['n_calls']}  missing={r['missing']}")

    n = len(rows)
    summary = {k: round(sum(float(r[k]) for r in rows) / n, 3)
               for k in ("tool_recall", "tool_precision", "exec_success", "exact_cover", "n_calls", "seconds")}
    print(f"\n{args.planner} planner over {n} tasks: {json.dumps(summary)}")
    if args.out:
        with open(args.out, "w") as f:
            json.dump({"planner": args.planner, "summary": summary, "tasks": rows}, f, indent=2)


if __name__ == "__main__":
    main()
