import io
import warnings
from contextlib import redirect_stdout, redirect_stderr
from typing import List, Tuple

import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator

def generate_final_marking_from_net(net: PetriNet) -> Marking:
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

    return {p: 1 for p in sinks}

def _quiet_call(fn, *args, quiet=True, **kwargs):
    if not quiet:
        return fn(*args, **kwargs)
    sink = io.StringIO()
    with redirect_stdout(sink), redirect_stderr(sink):
        return fn(*args, **kwargs)

def _run_metrics(log: EventLog, net: PetriNet, im: Marking, fm: Marking, quiet: bool = True):
    if fm is None or len(fm) == 0:
        fm = generate_final_marking_from_net(net)

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
        token_replay.apply, log, net, im, fm,
        parameters={"show_progress_bar": not quiet},
        quiet=quiet
    )
    if token_replay_result is None:
        raise RuntimeError("token_replay.apply returned None")

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
                trace_fitnesses.append(max(0.0, 1.0 - (missing + remaining) / denom))
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
            log, net, im, fm,
            variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN,
            parameters={"show_progress_bar": not quiet},
            quiet=quiet,
        )
    except Exception as e:
        print(f"Warning: Could not compute precision: {e}")
        precision = 0.0

    return avg_fitness, precision, trace_fitnesses

def compute_metrics(xes_path: str, pnml_path: str, quiet: bool = True) -> Tuple[float, float, List[float]]:
    log = _quiet_call(pm4py.read_xes, xes_path, quiet=quiet)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="the Petri net has been imported without a specified final marking.*",
            category=UserWarning,
        )
        net, im, fm = _quiet_call(pm4py.read_pnml, pnml_path, quiet=quiet)
    return _run_metrics(log, net, im, fm, quiet=quiet)

def compute_metrics_from_objects(
    log: EventLog,
    net: PetriNet,
    im: Marking,
    fm: Marking,
    quiet: bool = True,
) -> Tuple[float, float, List[float]]:
    return _run_metrics(log, net, im, fm, quiet=quiet)
