import glob
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'src'))

import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

from compute_fitness import generate_final_marking_from_net
from log_parser import XESLogParser
from synthesis_algorithm import MultiAgentPetriNetSynthesizer

def _ensure_labels(net):
    for t in net.transitions:
        try:
            cur_label = getattr(t, 'label', None) or getattr(t, 'name', None)
            if isinstance(cur_label, str) and cur_label.strip():
                continue
            tid = getattr(t, 'id', None) or str(getattr(t, 'name', None) or t)
            if tid:
                setattr(t, 'label', str(tid))
                setattr(t, 'name', str(tid))
        except Exception:
            pass

def _fitness_from_replay(tr_res):
    trace_fitnesses = []
    for r in tr_res:
        if isinstance(r, dict):
            if 'trace_fitness' in r:
                trace_fitnesses.append(float(r['trace_fitness']))
            elif 'fitness' in r:
                trace_fitnesses.append(float(r['fitness']))
            else:
                produced = float(r.get('produced_tokens', 0) or 0)
                missing = float(r.get('missing_tokens', 0) or 0)
                remaining = float(r.get('remaining_tokens', 0) or 0)
                denom = produced if produced > 0 else 1.0
                trace_fitnesses.append(
                    max(0.0, 1.0 - (missing + remaining) / denom)
                )
        else:
            try:
                trace_fitnesses.append(float(r))
            except Exception:
                trace_fitnesses.append(0.0)
    if not trace_fitnesses:
        return 0.0
    return sum(trace_fitnesses) / len(trace_fitnesses)

def compute_fitness_precision(log, net, im, fm):
    if fm is None or len(fm) == 0:
        fm = generate_final_marking_from_net(net)
    _ensure_labels(net)

    try:
        tr_res = token_replay.apply(log, net, im, fm)
        fitness = _fitness_from_replay(tr_res)
    except Exception as e:
        print(f"    * Warning: token-replay failed: {e}")
        fitness = 0.0

    try:
        try:
            variant = precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        except AttributeError:
            variant = precision_evaluator.Variants.ALIGN_ETCP
        precision = precision_evaluator.apply(log, net, im, fm, variant=variant)
    except Exception as e:
        print(f"    * Warning: precision failed: {e}")
        precision = 0.0

    return fitness, precision

def _locate_log(data_dir, ds):
    data_path = os.path.join(data_dir, ds)
    files = glob.glob(os.path.join(data_path, "*_init_log_enriched.xes"))
    if not files:
        files = glob.glob(os.path.join(data_path, "*.xes"))
    files = [
        f for f in files
        if "__MACOSX" not in f and not os.path.basename(f).startswith('.')
    ]
    return files[0] if files else None

def _log_stats(log):
    n_traces = 0
    n_events = 0
    activities = set()
    agents = set()
    for trace in log:
        n_traces += 1
        for event in trace:
            n_events += 1
            act = event.get("concept:name")
            if act:
                activities.add(act)
            resource = event.get("org:resource")
            if resource:
                for a in str(resource).split("+"):
                    a = a.strip()
                    if a and a != "Unknown":
                        agents.add(a)
    return n_traces, n_events, len(activities), len(agents)

def compute_for_dataset(ds, data_dir="data"):
    result = {
        "Pattern": ds,
        "Traces": "-",
        "Events": "-",
        "Activities": "-",
        "Agents": "-",
        "Channels": "-",
        "Fitness": "-",
        "Precision": "-",
        "Note": "",
    }

    log_file = _locate_log(data_dir, ds)
    if not log_file:
        result["Note"] = "no xes log"
        return result
    print(f"\n--- {ds} ---")
    print(f"    log: {log_file}")

    try:
        log = XESLogParser.load_log(log_file)
    except Exception as e:
        result["Note"] = f"log load error: {e}"
        print(f"    [ERROR] {result['Note']}")
        return result

    n_traces, n_events, n_activities, n_agents = _log_stats(log)
    result.update({
        "Traces": n_traces,
        "Events": n_events,
        "Activities": n_activities,
        "Agents": n_agents,
    })
    print(
        f"    traces={n_traces}, events={n_events}, "
        f"activities={n_activities}, agents={n_agents}"
    )

    try:
        synthesizer = MultiAgentPetriNetSynthesizer(log)
        sys_net, sys_im, sys_fm = synthesizer.run()

        n_channels = sum(
            1 for p in sys_net.places if str(p.name).startswith("chan_")
        )
        fitness, precision = compute_fitness_precision(
            log, sys_net, sys_im, sys_fm
        )
        result.update({
            "Channels": n_channels,
            "Fitness": f"{fitness:.4f}",
            "Precision": f"{precision:.4f}",
        })
        print(
            f"    channels={n_channels}, "
            f"fitness={fitness:.4f}, precision={precision:.4f}"
        )
    except Exception as e:
        result["Note"] = f"synthesis/eval error: {e}"
        print(f"    [ERROR] {result['Note']}")
        traceback.print_exc()

    return result

def _dataset_sort_key(name):
    try:
        return (0, int(name.split('-')[1]))
    except Exception:
        return (1, name)

def run_all(data_dir="data", output_md="metrics_all_datasets.md"):
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return

    datasets = sorted(
        [
            d for d in os.listdir(data_dir)
            if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')
        ],
        key=_dataset_sort_key,
    )
    if not datasets:
        print("No datasets found.")
        return

    print(f"Found {len(datasets)} dataset(s): {', '.join(datasets)}")

    results = [compute_for_dataset(ds, data_dir) for ds in datasets]

    col_order = [
        "Pattern", "Traces", "Events", "Activities", "Agents",
        "Channels", "Fitness", "Precision", "Note",
    ]
    widths = {
        c: max(len(c), max(len(str(r[c])) for r in results))
        for c in col_order
    }
    border = "+".join("-" * (widths[c] + 2) for c in col_order)
    print("\n" + border)
    print(
        "|" + "|".join(f" {c.ljust(widths[c])} " for c in col_order) + "|"
    )
    print(border)
    for r in results:
        print(
            "|" + "|".join(
                f" {str(r[c]).ljust(widths[c])} " for c in col_order
            ) + "|"
        )
    print(border)

    totals = {
        "traces": sum(int(r["Traces"]) for r in results if str(r["Traces"]).isdigit()),
        "events": sum(int(r["Events"]) for r in results if str(r["Events"]).isdigit()),
        "datasets": len(results),
    }
    print(
        f"\nTotals: {totals['datasets']} datasets, "
        f"{totals['traces']} traces, {totals['events']} events."
    )

    try:
        with open(output_md, "w", encoding="utf-8") as f:
            f.write("# Metrics across all datasets\n\n")
            f.write(
                "| Pattern | Traces | Events | Activities | Agents | "
                "Channels | Fitness | Precision | Note |\n"
            )
            f.write(
                "|---------|--------|--------|------------|--------|"
                "----------|---------|-----------|------|\n"
            )
            for r in results:
                f.write(
                    f"| {r['Pattern']} | {r['Traces']} | {r['Events']} | "
                    f"{r['Activities']} | {r['Agents']} | {r['Channels']} | "
                    f"{r['Fitness']} | {r['Precision']} | {r['Note']} |\n"
                )
            f.write(
                f"\n**Totals:** {totals['datasets']} datasets, "
                f"{totals['traces']} traces, {totals['events']} events.\n"
            )
        print(f"Markdown report written to: {output_md}")
    except Exception as e:
        print(f"Could not write markdown report: {e}")

if __name__ == "__main__":
    run_all()
