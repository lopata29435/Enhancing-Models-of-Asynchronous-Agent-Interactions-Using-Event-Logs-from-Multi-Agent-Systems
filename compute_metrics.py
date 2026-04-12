import argparse
import sys
import os
import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

def generate_final_marking_from_net(net):
    try:
        sinks = [p for p in net.places if len(getattr(p, "out_arcs", [])) == 0]
    except Exception:
        sinks = []

    if not sinks:
        try:
            sinks = [p for p in net.places if all(getattr(a, "target", None) is None for a in getattr(p, "out_arcs", []))]
        except Exception:
            sinks = []

    if not sinks:
        return {}

    marking = {p: 1 for p in sinks}
    return marking

def _quiet_call(fn, *args, quiet=True, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)

def compute_metrics(xes_path: str, pnml_path: str, quiet: bool = True):
    log = _quiet_call(pm4py.read_xes, xes_path, quiet=quiet)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the Petri net has been imported without a specified final marking.*",
            category=UserWarning,
        )
        net, initial_marking, final_marking = _quiet_call(pm4py.read_pnml, pnml_path, quiet=quiet)
    if final_marking is None:
        final_marking = generate_final_marking_from_net(net)

    for t in net.transitions:
        try:
            cur_label = None
            if hasattr(t, 'label'):
                cur_label = getattr(t, 'label')
            if not cur_label and hasattr(t, 'name'):
                cur_label = getattr(t, 'name')
            if isinstance(cur_label, str) and cur_label.strip():
                continue
            tid = None
            if hasattr(t, 'id'):
                tid = getattr(t, 'id')
            else:
                for k, v in t.__dict__.items():
                    if k.lower().endswith('id') and isinstance(v, str):
                        tid = v
                        break
            if not tid:
                try:
                    tid = str(getattr(t, 'name', None) or t)
                except Exception:
                    tid = None
            if tid:
                try:
                    setattr(t, 'label', str(tid))
                except Exception:
                    pass
                try:
                    setattr(t, 'name', str(tid))
                except Exception:
                    pass
        except Exception:
            continue

    if not quiet:
        print("      -> Starting token replay...", flush=True)

    token_replay_result = _quiet_call(
        token_replay.apply, log, net, initial_marking, final_marking,
        parameters={"show_progress_bar": not quiet},
        quiet=quiet
    )
    if token_replay_result is None:
        raise RuntimeError("token_replay.apply returned None (possibly incompatible net/markings)")
    
    trace_fitnesses = []
    for r in token_replay_result:
        if isinstance(r, dict):
            if 'trace_fitness' in r:
                trace_fitnesses.append(float(r['trace_fitness']))
            elif 'fitness' in r:
                trace_fitnesses.append(float(r['fitness']))
            else:
                missing = float(r.get('missing_tokens', 0) or 0)
                remaining = float(r.get('remaining_tokens', 0) or 0)
                produced = float(r.get('produced_tokens', 0) or 0)
                denom = produced if produced > 0 else 1.0
                tf = max(0.0, 1.0 - (missing + remaining) / denom)
                trace_fitnesses.append(tf)
        else:
            try:
                trace_fitnesses.append(float(r))
            except Exception:
                trace_fitnesses.append(0.0)

    avg_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0

    if not quiet:
        print("      -> Starting precision calculation (ETC)...", flush=True)

    try:
        precision = _quiet_call(
            precision_evaluator.apply,
            log, net, initial_marking, final_marking,
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN,
            parameters={"show_progress_bar": not quiet},
            quiet=quiet,
        )
    except Exception as e:
        print(f"Warning: Could not compute precision: {e}")
        precision = 0.0

    return avg_fitness, precision, trace_fitnesses

def select_dataset(base_dir="output"):
    if not os.path.exists(base_dir):
        print(f"Directory '{base_dir}' not found. Please run the synthesis algorithm first.")
        return None
    
    datasets = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and not d.startswith('.')]
    datasets.sort(key=lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0)
    
    if not datasets:
        print(f"No datasets found in {base_dir}.")
        return None
        
    print("\n" + "="*40)
    print("Available datasets for evaluation:")
    for i, ds in enumerate(datasets):
        print(f"{i + 1}. {ds}")
    print("="*40)
        
    while True:
        choice = input("\nSelect a dataset number to evaluate (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def evaluate_dataset(dataset_name):
    data_path = os.path.join("data", dataset_name)
    output_path = os.path.join("output", dataset_name)

    log_file = os.path.join(data_path, f"{dataset_name}_init_log.xes")
    ref_file = os.path.join(data_path, f"{dataset_name}_ref_model.pnml")
    mined_file = os.path.join(output_path, f"{dataset_name}_composition_mined.pnml")

    result = {
        "dataset": dataset_name,
        "log_file": log_file,
        "ref_model": ref_file,
        "mined_model": mined_file,
        "ref_fitness": None,
        "ref_precision": None,
        "mined_fitness": None,
        "mined_precision": None,
        "ref_error": None,
        "mined_error": None,
    }

    if not os.path.exists(log_file):
        result["ref_error"] = f"Log file not found: {log_file}"
        result["mined_error"] = result["ref_error"]
        return result

    if not os.path.exists(ref_file):
        result["ref_error"] = f"Reference model not found: {ref_file}"
    else:
        try:
            avg, precision, _ = compute_metrics(log_file, ref_file, quiet=True)
            result["ref_fitness"] = avg
            result["ref_precision"] = precision
        except Exception as e:
            result["ref_error"] = str(e)

    if not os.path.exists(mined_file):
        result["mined_error"] = f"Synthesized model not found: {mined_file}"
    else:
        try:
            avg, precision, _ = compute_metrics(log_file, mined_file, quiet=True)
            result["mined_fitness"] = avg
            result["mined_precision"] = precision
        except Exception as e:
            result["mined_error"] = str(e)

    return result

def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--interactive":
        parser = argparse.ArgumentParser(description="Compute fitness between XES log and PNML model")
        parser.add_argument("log", help="Path to XES event log")
        parser.add_argument("model", help="Path to PNML model")
        parser.add_argument("--diagnose", action="store_true", help="Print diagnostics")
        parser.add_argument("--show-traces", action="store_true", help="Print per-trace fitness values")
        args = parser.parse_args()

        try:
            avg, precision, traces = compute_metrics(args.log, args.model, quiet=not args.diagnose)
            print(f"Average fitness: {avg:.6f}")
            print(f"Precision: {precision:.6f}")
        except Exception as e:
            print("Error during computation:", e)
            sys.exit(2)
        return

    while True:
        dataset = select_dataset("data")
        if dataset:
            summary = evaluate_dataset(dataset)
            print("\n" + "="*50)
            print(f"Dataset: {summary['dataset']}")
            if summary["ref_error"]:
                print(f"Reference: ERROR - {summary['ref_error']}")
            else:
                print(
                    f"Reference: fitness={summary['ref_fitness']:.6f}, "
                    f"precision={summary['ref_precision']:.6f}"
                )
            if summary["mined_error"]:
                print(f"Synthesized: ERROR - {summary['mined_error']}")
            else:
                print(
                    f"Synthesized: fitness={summary['mined_fitness']:.6f}, "
                    f"precision={summary['mined_precision']:.6f}"
                )
            print("="*50)
        else:
            print("Exiting...")
            break

if __name__ == "__main__":
    import os
    main()
