#!/usr/bin/env python

import argparse
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import List, Optional, Set, Tuple

import pm4py
from pm4py.objects.log.obj import EventLog
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.convert import convert_to_event_log

from src.synthesis_algorithm import localize_acyclic_interaction, localize_cyclic_interaction
from src.log_parser import XESLogParser
from src.async_agents import (
    _compose_with_channels,
    _mine_perfect_fit_petri,
)
from src.transitive_reduction import (
    build_precedence_map,
    remove_transitive_channels,
)
from src.disconnected_agents import compose_disconnected
from src.random_channel_removal import iterative_random_removal
from utils.hybrid_metrics import compute_hybrid_metrics
from compute_metrics import compute_metrics as compute_metrics_from_files

DATA_DIR = Path("data")
OUTPUT_DIR = Path("output")
AGENT_ATTR = "org:resource"

def _load_log(xes_path: Path, quiet: bool = False) -> EventLog:
    import pm4py.util.constants
    pm4py.util.constants.SHOW_PROGRESS_BAR = False
    
    log_or_df = pm4py.read_xes(str(xes_path))
    if not isinstance(log_or_df, EventLog):
        return convert_to_event_log(log_or_df)
    return log_or_df

def _find_xes(dataset_dir: Path) -> Optional[Path]:
    for f in sorted(dataset_dir.iterdir()):
        if f.suffix.lower() == ".xes" and "hybrid" not in f.name.lower():
            return f
    return None

def project_log_for_agent(log: EventLog, agent_attr: str, agent_name: str) -> EventLog:
    from pm4py.objects.log.obj import Trace
    new_log = EventLog()
    for trace in log:
        new_trace = Trace()
        for event in trace:
            agents_in_event = XESLogParser.identify_agents(event)
            if agent_name in agents_in_event:
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log

def enrich_unassigned_send_receive(log: EventLog, agent_attr: str = AGENT_ATTR) -> int:
    import re
    import math

    send_owner: dict = {}
    receive_events_missing: list = []

    def _has_resource(ev) -> bool:
        val = ev.get(agent_attr, None)
        if val is None:
            return False
        if isinstance(val, float) and math.isnan(val):
            return False
        s = str(val).strip()
        return bool(s) and s.lower() not in {"unknown", "nan", "none"}

    base_re = re.compile(r"^([^!?]+)([!?])(.*)$")

    for trace in log:
        for ev in trace:
            name = str(ev.get("concept:name", ""))
            if not name:
                continue
            m = base_re.match(name)
            if not m:
                continue
            base, kind, _suffix = m.group(1), m.group(2), m.group(3)
            if _has_resource(ev):
                if kind == "!":
                    agents = XESLogParser.identify_agents(ev)
                    for a in agents:
                        send_owner.setdefault(base, set()).add(a)
            else:
                receive_events_missing.append((ev, base, kind))

    if not receive_events_missing:
        return 0

    all_known_agents = set()
    for trace in log:
        for ev in trace:
            all_known_agents.update(XESLogParser.identify_agents(ev))
    if len(all_known_agents) < 2:
        return 0

    updated = 0
    for ev, base, kind in receive_events_missing:
        if kind != "?":
            continue
        owners = send_owner.get(base)
        if not owners or len(owners) != 1:
            continue
        sender = next(iter(owners))
        receivers = [a for a in all_known_agents if a != sender]
        if len(receivers) != 1:
            continue
        ev[agent_attr] = receivers[0]
        updated += 1
    return updated

def _detect_pattern(dataset_name: str) -> Optional[int]:
    m = re.search(r"IP-?(\d+)", dataset_name, re.IGNORECASE)
    if m:
        n = int(m.group(1))
        return n if 1 <= n <= 7 else None
    return None

def _fmt(value, decimals=4):
    if value is None:
        return "-"
    return f"{value:.{decimals}f}"

def wrap_with_global_markings(
    net: PetriNet, im: Marking, fm: Marking,
) -> Tuple[PetriNet, Marking, Marking]:
    p_start = PetriNet.Place("global_start")
    p_end = PetriNet.Place("global_end")
    net.places.add(p_start)
    net.places.add(p_end)

    tau_start = PetriNet.Transition("tau_start", None)
    tau_end = PetriNet.Transition("tau_end", None)
    net.transitions.add(tau_start)
    net.transitions.add(tau_end)

    petri_utils.add_arc_from_to(p_start, tau_start, net)
    for p in im:
        petri_utils.add_arc_from_to(tau_start, p, net)

    petri_utils.add_arc_from_to(tau_end, p_end, net)
    for p in fm:
        petri_utils.add_arc_from_to(p, tau_end, net)

    return net, Marking({p_start: 1}), Marking({p_end: 1})

def eval_net_via_pnml(
    net, im, fm, xes_path: str, quiet: bool = True,
) -> Tuple[float, float]:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".pnml")
    os.close(tmp_fd)
    try:
        pm4py.write_pnml(net, im, fm, tmp_path)
        fit, prec, _ = compute_metrics_from_files(xes_path, tmp_path, quiet=quiet)
        return fit, prec
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

def detect_agents_and_events(log: EventLog):
    traces: List[List[str]] = []
    for trace in log:
        events = [str(e.get("concept:name", "")) for e in trace]
        events = [e for e in events if e]
        if events:
            traces.append(events)

    agent_events_set: dict = {}
    first_seen: dict = {}
    global_idx = 0

    for trace in log:
        for event in trace:
            event_name = str(event.get("concept:name", ""))
            if not event_name:
                continue
            agents_list = XESLogParser.identify_agents(event)
            if event_name not in first_seen:
                first_seen[event_name] = global_idx
            global_idx += 1

            for agent in agents_list:
                if not agent or agent.lower() == "nan":
                    continue
                agent_events_set.setdefault(agent, set()).add(event_name)

    ordered_agents = sorted(agent_events_set.keys())
    agent_events: dict = {}
    for agent in ordered_agents:
        evts = sorted(
            agent_events_set[agent],
            key=lambda x: first_seen.get(x, float("inf")),
        )
        agent_events[agent] = evts

    return agent_events, traces

def extract_channels(
    traces: List[List[str]],
    a1_events: List[str],
    a2_events: List[str],
    agent1: str,
    agent2: str,
    is_cyclic: bool = False,
):
    a1_set = set(a1_events)
    a2_set = set(a2_events)
    channels = []

    def _classify_acyclic(raw_acyclic):
        for sender_act, receiver_act in raw_acyclic:
            if sender_act in a1_set and receiver_act in a2_set:
                channels.append((sender_act, receiver_act, "<"))
            elif sender_act in a2_set and receiver_act in a1_set:
                channels.append((receiver_act, sender_act, ">"))
            else:
                print(f"  [WARNING] Cannot classify channel: "
                      f"{sender_act} -> {receiver_act}")

    def _classify_cyclic(raw_cyclic):
        for ch in raw_cyclic:
            sender_act = ch["sender"]
            receiver_act = ch["receiver"]
            if sender_act in a1_set and receiver_act in a2_set:
                channels.append((sender_act, receiver_act, "<"))
            elif sender_act in a2_set and receiver_act in a1_set:
                channels.append((receiver_act, sender_act, ">"))
            else:
                print(f"  [WARNING] Cannot classify cyclic channel: "
                      f"{sender_act} -> {receiver_act}")

    if not is_cyclic:
        raw = localize_acyclic_interaction(traces, a1_events, a2_events)
        _classify_acyclic(raw)
    else:
        all_events = set(a1_events) | set(a2_events)
        cyclic_events = set()
        for trace in traces:
            from collections import Counter
            counts = Counter(trace)
            for evt, cnt in counts.items():
                if cnt > 1 and evt in all_events:
                    cyclic_events.add(evt)
        acyclic_events = all_events - cyclic_events

        a1_acyclic = [e for e in a1_events if e in acyclic_events]
        a2_acyclic = [e for e in a2_events if e in acyclic_events]
        a1_cyclic = [e for e in a1_events if e in cyclic_events]
        a2_cyclic = [e for e in a2_events if e in cyclic_events]

        print(f"  [IP-7] Acyclic events: a1={len(a1_acyclic)}, a2={len(a2_acyclic)}")
        print(f"  [IP-7] Cyclic events:  a1={len(a1_cyclic)}, a2={len(a2_cyclic)}")

        if a1_acyclic and a2_acyclic:
            raw_acyclic = localize_acyclic_interaction(
                traces, a1_acyclic, a2_acyclic,
            )
            _classify_acyclic(raw_acyclic)

        if a1_cyclic and a2_cyclic:
            raw_cyclic = localize_cyclic_interaction(
                traces, a1_cyclic, a2_cyclic,
            )
            _classify_cyclic(raw_cyclic)

    return channels

def validate_channels(
    channels: list,
    traces: List[List[str]],
    a1_events: List[str],
    a2_events: List[str],
    quiet: bool = False,
) -> list:
    valid = []
    removed = []

    for ch in channels:
        a1_act, a2_act, rel = ch
        if rel == "<":
            sender, receiver = a1_act, a2_act
        else:
            sender, receiver = a2_act, a1_act

        ok = True
        for trace in traces:
            s_in = sender in trace
            r_in = receiver in trace
            if s_in and not r_in:
                ok = False
                break
            if r_in and not s_in:
                ok = False
                break

        if ok:
            valid.append(ch)
        else:
            removed.append((ch, sender, receiver))

    if removed:
        for ch, s, r in removed:
            if not quiet:
                print(f"    [FILTER] Removed {s}->{r}: "
                     f"co-occurrence not guaranteed")

    return valid

def run_pipeline(
    dataset_name: str,
    verbose: bool = True,
    clean_isolated: bool = True,
    apply_tr: bool = True,
    hr_thresh_ag_prec: float = 0.05,
    hr_thresh_diff: float = -0.01,
    pattern: int = None,
    output_dir: Path = None,
    quiet: bool = False,
) -> Optional[dict]:
    dataset_dir = DATA_DIR / dataset_name
    xes_path = _find_xes(dataset_dir)
    if xes_path is None:
        if not quiet:
            print(f"[ERROR] No .xes log file found in {dataset_dir}")
        return None

    if pattern is None:
        pattern = _detect_pattern(dataset_name)

    def _log(msg: str = ""):
        if not quiet:
            print(msg)

    if pattern is not None:
        _log(f"======================================================================")
        _log(f"  8-STEP PIPELINE  —  IP-{pattern} ({dataset_name})")
        _log(f"======================================================================")
        _log(f"  Log file : {xes_path.name}")
        _log(f"  Pattern  : {pattern}")
    else:
        _log(f"======================================================================")
        _log(f"  8-STEP PIPELINE  —  {dataset_name}")
        _log(f"======================================================================")

    log = _load_log(xes_path)

    fixed = enrich_unassigned_send_receive(log, AGENT_ATTR)
    if fixed and not quiet:
        print(f"  [enrichment] assigned org:resource to {fixed} unresourced events")

    ref_pnml = xes_path.with_name(xes_path.stem.replace("_init_log", "_ref_model") + ".pnml")

    ref_fit, ref_prec = None, None
    if ref_pnml.exists():
        try:
            ref_fit, ref_prec, _ = compute_metrics_from_files(
                str(xes_path), str(ref_pnml), quiet=True,
            )
            _log(f"  Reference : fitness={_fmt(ref_fit)}, precision={_fmt(ref_prec)}")
        except Exception as e:
            _log(f"  Reference : error — {e}")

    _log(f"\n{'─'*70}")
    _log("[Step 1] Algorithm run — Inductive Miner per agent + "
         "matrix minimum localization")
    _log(f"{'─'*70}")

    agent_events, traces = detect_agents_and_events(log)
    agents = list(agent_events.keys())

    if len(agents) < 2:
        _log(f"[ERROR] Need at least 2 agents, found {len(agents)}")
        return None

    agent1, agent2 = agents[0], agents[1]
    a1_events = agent_events[agent1]
    a2_events = agent_events[agent2]

    _log(f"  Agent 1 : {agent1!r}  ({len(a1_events)} events)")
    _log(f"  Agent 2 : {agent2!r}  ({len(a2_events)} events)")

    _log("  [1.1] Projecting log and mining local Petri nets "
         "(Inductive Miner)...")
    log_a = project_log_for_agent(log, AGENT_ATTR, agent1)
    log_b = project_log_for_agent(log, AGENT_ATTR, agent2)
    pn_a, im_a, fm_a = _mine_perfect_fit_petri(log_a)
    pn_b, im_b, fm_b = _mine_perfect_fit_petri(log_b)
    _log(f"        Net(Agent1) : |P|={len(pn_a.places)}, "
         f"|T|={len(pn_a.transitions)}, |F|={len(pn_a.arcs)}")
    _log(f"        Net(Agent2) : |P|={len(pn_b.places)}, "
         f"|T|={len(pn_b.transitions)}, |F|={len(pn_b.arcs)}")

    _log("  [1.2] Computing matrix and minimum (channel candidates)...")
    is_cyclic = (pattern is not None and pattern >= 7)
    channels = extract_channels(
        traces, a1_events, a2_events, agent1, agent2, is_cyclic=is_cyclic,
    )

    _log(f"        Channels from minimum : {len(channels)}")
    raw_channels_list = list(channels)

    channels = validate_channels(
        channels, traces, a1_events, a2_events, quiet=quiet,
    )
    _log(f"        Channels after co-occurrence check : {len(channels)}")

    if verbose:
        for a, b, r in channels:
            direction = f"{a} -> {b}" if r == "<" else f"{b} -> {a}"
            _log(f"          * {direction}  (rel={r})")

    _log(f"\n{'─'*70}")
    _log("[Step 2] Composition — wiring Agent1 and Agent2 nets with "
         "all minimum channels")
    _log(f"{'─'*70}")

    base_net, base_im, base_fm = _compose_with_channels(
        pn_a, im_a, fm_a, pn_b, im_b, fm_b, channels,
    )
    base_net, base_im, base_fm = wrap_with_global_markings(
        base_net, base_im, base_fm,
    )

    fit_base, prec_base = eval_net_via_pnml(
        base_net, base_im, base_fm, str(xes_path),
    )
    _log(f"  Channels in base model : {len(channels)}")
    _log(f"  Fitness_base           : {_fmt(fit_base)}")
    _log(f"  Precision_base         : {_fmt(prec_base)}")

    _log(f"\n{'─'*70}")
    _log("[Step 3] Transitive channel removal — structural redundancy analysis")
    _log(f"{'─'*70}")

    prec_map1 = build_precedence_map(log, AGENT_ATTR, agent1)
    prec_map2 = build_precedence_map(log, AGENT_ATTR, agent2)

    kept_channels, removed_pairs = remove_transitive_channels(
        channels, agent1, agent2, prec_map1, prec_map2,
    )

    _log(f"  Removed (transitive) : {len(removed_pairs)}")
    for redundant, witness in removed_pairs:
        r_dir = (f"{redundant[0]}->{redundant[1]}" if redundant[2] == "<"
                 else f"{redundant[1]}->{redundant[0]}")
        w_dir = (f"{witness[0]}->{witness[1]}" if witness[2] == "<"
                 else f"{witness[1]}->{witness[0]}")
        _log(f"    x {r_dir}  implied by {w_dir}")
    _log(f"  Remaining channels   : {len(kept_channels)}")

    if removed_pairs:
        _log(f"\n{'─'*70}")
        _log("[Step 4] Metrics after transitive cleanup")
        _log(f"{'─'*70}")

        clean_net, clean_im, clean_fm = _compose_with_channels(
            pn_a, im_a, fm_a, pn_b, im_b, fm_b, kept_channels,
        )
        clean_net, clean_im, clean_fm = wrap_with_global_markings(
            clean_net, clean_im, clean_fm,
        )

        fit_trans, prec_trans = eval_net_via_pnml(
            clean_net, clean_im, clean_fm, str(xes_path),
        )
        _log(f"  Channels remaining       : {len(kept_channels)}")
        _log(f"  Fitness_transitive       : {_fmt(fit_trans)}")
        _log(f"  Precision_transitive     : {_fmt(prec_trans)}")
    else:
        _log(f"\n{'─'*70}")
        _log("[Step 4] Skipped — no transitive channels removed")
        _log(f"{'─'*70}")
        clean_net, clean_im, clean_fm = base_net, base_im, base_fm
        fit_trans, prec_trans = fit_base, prec_base

    _log(f"\n{'─'*70}")
    _log("[Step 5] Hybrid model metrics — R1~A2 and A1~R2")
    _log(f"{'─'*70}")

    prec_r1_a2 = None
    prec_a1_r2 = None
    min_ag_p = 0.0

    if pattern is not None and 1 <= pattern <= 7:
        try:
            hybrid_results = compute_hybrid_metrics(
                log=log,
                channels=kept_channels,
                agent1=agent1,
                agent2=agent2,
                pattern=pattern,
                real_nets={
                    "a1": (pn_a, im_a, fm_a),
                    "a2": (pn_b, im_b, fm_b),
                },
                agent_attr=AGENT_ATTR,
            )
            prec_r1_a2 = hybrid_results["prec_R1_A2"]
            prec_a1_r2 = hybrid_results["prec_A1_R2"]
            fit_r1_a2 = hybrid_results["fit_R1_A2"]
            fit_a1_r2 = hybrid_results["fit_A1_R2"]
            min_ag_p = min(prec_r1_a2, prec_a1_r2)

            _log(f"  Precision(R1~A2)  : {_fmt(prec_r1_a2)}")
            _log(f"  Fitness(R1~A2)    : {_fmt(fit_r1_a2)}")
            _log(f"  Precision(A1~R2)  : {_fmt(prec_a1_r2)}")
            _log(f"  Fitness(A1~R2)    : {_fmt(fit_a1_r2)}")
            _log(f"  min_ag_p          : {_fmt(min_ag_p)}")
        except Exception as exc:
            _log(f"  [WARNING] Hybrid metrics failed: {exc}")
    else:
        _log(f"  [SKIP] Pattern {pattern or 'N/A'} outside 1-7")

    _log(f"\n{'─'*70}")
    _log("[Step 6] Disconnected agents — no channels, lower bound precision")
    _log(f"{'─'*70}")

    disc_net, disc_im, disc_fm = compose_disconnected(
        pn_a, im_a, fm_a, pn_b, im_b, fm_b,
    )
    disc_net, disc_im, disc_fm = wrap_with_global_markings(
        disc_net, disc_im, disc_fm,
    )

    fit_disc, prec_disc = eval_net_via_pnml(
        disc_net, disc_im, disc_fm, str(xes_path),
    )
    _log(f"  Fitness_disconnected   : {_fmt(fit_disc)}")
    _log(f"  Precision_disconnected : {_fmt(prec_disc)}")

    _log(f"\n{'─'*70}")
    _log("[Step 7] Heuristic channel removal — iterative with precision "
          "thresholds")
    _log(f"{'─'*70}")
    _log(f"  Thresholds: min_ag_p={_fmt(min_ag_p)}, "
          f"prec_disc={_fmt(prec_disc)}")

    def _metrics_fn(net, im, fm):
        return eval_net_via_pnml(net, im, fm, str(xes_path))

    n_before, n_removed, removed_names, kept_names = iterative_random_removal(
        net=clean_net,
        im=clean_im,
        fm=clean_fm,
        metrics_fn=_metrics_fn,
        min_ag_p=min_ag_p,
        prec_disconnected=prec_disc,
        seed=42,
        quiet=not verbose,
    )

    _log(f"  Channels before heuristic : {n_before}")
    _log(f"  Channels removed          : {n_removed}")
    _log(f"  Channels remaining        : {n_before - n_removed}")
    if verbose and removed_names:
        for name in removed_names:
            _log(f"    x {name}")

    _log(f"\n{'─'*70}")
    _log("[Step 8] Final summary")
    _log(f"{'─'*70}")

    out_dir = OUTPUT_DIR / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_pnml = out_dir / f"{dataset_name}_pipeline_optimized.pnml"
    pm4py.write_pnml(clean_net, clean_im, clean_fm, str(out_pnml))

    fit_final, prec_final = eval_net_via_pnml(
        clean_net, clean_im, clean_fm, str(xes_path),
    )
    n_final = len([p for p in clean_net.places if p.name.startswith("chan_")])

    _log()
    _log(f"  {'Metric':<35} {'Value':>10}")
    _log(f"  {'─'*46}")
    if ref_fit is not None:
        _log(f"  {'Ref Fitness':<35} {_fmt(ref_fit):>10}")
        _log(f"  {'Ref Precision':<35} {_fmt(ref_prec):>10}")
    _log(f"  {'Candidate channels (Step 1)':<35} {len(channels):>10}")
    _log(f"  {'Fitness_base (Step 2)':<35} {_fmt(fit_base):>10}")
    _log(f"  {'Precision_base (Step 2)':<35} {_fmt(prec_base):>10}")
    _log(f"  {'Removed transitive (Step 3)':<35} {len(removed_pairs):>10}")
    _log(f"  {'Fitness_trans (Step 4)':<35} {_fmt(fit_trans):>10}")
    _log(f"  {'Precision_trans (Step 4)':<35} {_fmt(prec_trans):>10}")
    _log(f"  {'Precision(R1~A2) (Step 5)':<35} {_fmt(prec_r1_a2):>10}")
    _log(f"  {'Precision(A1~R2) (Step 5)':<35} {_fmt(prec_a1_r2):>10}")
    _log(f"  {'min_ag_p (Step 5)':<35} {_fmt(min_ag_p):>10}")
    _log(f"  {'Fitness_disconnected (Step 6)':<35} {_fmt(fit_disc):>10}")
    _log(f"  {'Precision_disconnected (Step 6)':<35} {_fmt(prec_disc):>10}")
    _log(f"  {'Removed heuristic (Step 7)':<35} {n_removed:>10}")
    _log(f"  {'Final channels (Step 8)':<35} {n_final:>10}")
    _log(f"  {'Final Fitness':<35} {_fmt(fit_final):>10}")
    _log(f"  {'Final Precision':<35} {_fmt(prec_final):>10}")
    _log(f"  {'─'*46}")

    hyp_ag = prec_final >= min_ag_p if min_ag_p > 0 else True
    hyp_disc = prec_final >= prec_disc
    _log(f"  Final Prec >= min_ag_p?           {'YES' if hyp_ag else 'NO'}")
    _log(f"  Final Prec >= Prec_disconnected?  {'YES' if hyp_disc else 'NO'}")
    _log(f"  Optimized model saved -> {out_pnml}")
    _log(f"{'='*70}\n")

    return {
        "dataset": dataset_name,
        "raw_candidates": len(raw_channels_list),
        "candidates": len(channels),
        "removed_transitive": len(removed_pairs),
        "removed_heuristic": n_removed,
        "final_channels": n_final,
        "fit_base": fit_base,
        "prec_base": prec_base,
        "fit_trans": fit_trans,
        "prec_trans": prec_trans,
        "prec_R1_A2": prec_r1_a2,
        "prec_A1_R2": prec_a1_r2,
        "min_ag_p": min_ag_p,
        "fit_disc": fit_disc,
        "prec_disc": prec_disc,
        "fit_final": fit_final,
        "prec_final": prec_final,
    }

def _build_single_report(r: dict) -> str:
    lines = [
        f"# Pipeline Report — {r['dataset']}\n",
        "## Summary\n",
        "| Metric | Value |",
        "|:---|---:|",
        f"| Candidate channels | {r['candidates']} |",
        f"| Removed transitive | {r['removed_transitive']} |",
        f"| Removed heuristic | {r['removed_heuristic']} |",
        f"| **Final channels** | **{r['final_channels']}** |",
        "",
        "## Metrics\n",
        "| Step | Fitness | Precision |",
        "|:---|---:|---:|",
        f"| Base (all candidates) | {_fmt(r['fit_base'])} | {_fmt(r['prec_base'])} |",
        f"| After transitive removal | {_fmt(r['fit_trans'])} | {_fmt(r['prec_trans'])} |",
        f"| Disconnected (no channels) | {_fmt(r['fit_disc'])} | {_fmt(r['prec_disc'])} |",
        f"| **Final (optimized)** | **{_fmt(r['fit_final'])}** | **{_fmt(r['prec_final'])}** |",
        "",
        "## Hybrid Metrics\n",
        "| Model | Precision |",
        "|:---|---:|",
        f"| R1~A2 | {_fmt(r['prec_R1_A2'])} |",
        f"| A1~R2 | {_fmt(r['prec_A1_R2'])} |",
        f"| **min_ag_p** | **{_fmt(r['min_ag_p'])}** |",
        "",
    ]
    return "\n".join(lines)

def _build_combined_report(results: list) -> str:
    lines = [
        "# Pipeline Report — All Datasets\n",
        "## Channel Optimization\n",
        "| Dataset | Raw Cands | Candidates | Trans. Rem. | Heur. Rem. | Final |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {r.get('raw_candidates', '-')} | {r['candidates']} "
            f"| {r['removed_transitive']} | {r['removed_heuristic']} "
            f"| {r['final_channels']} |"
        )

    lines += [
        "",
        "## System Metrics\n",
        "| Dataset | Fit_base | Prec_base | Prec_trans "
        "| Prec_disc | **Prec_final** |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {_fmt(r['fit_base'])} "
            f"| {_fmt(r['prec_base'])} | {_fmt(r['prec_trans'])} "
            f"| {_fmt(r['prec_disc'])} | **{_fmt(r['prec_final'])}** |"
        )

    lines += [
        "",
        "## Hybrid Metrics\n",
        "| Dataset | Prec(R1~A2) | Prec(A1~R2) | min_ag_p |",
        "|:---|---:|---:|---:|",
    ]
    for r in results:
        lines.append(
            f"| {r['dataset']} | {_fmt(r['prec_R1_A2'])} "
            f"| {_fmt(r['prec_A1_R2'])} | {_fmt(r['min_ag_p'])} |"
        )

    lines += [
        "",
        "## Hypothesis Check\n",
        "| Dataset | Prec_final ≥ min_ag_p | Prec_final > Prec_disc |",
        "|:---|:---:|:---:|",
    ]
    for r in results:
        h1 = "✅" if (r['min_ag_p'] <= 0 or r['prec_final'] >= r['min_ag_p']) else "❌"
        h2 = "✅" if r['prec_final'] > r['prec_disc'] else "❌"
        lines.append(f"| {r['dataset']} | {h1} | {h2} |")

    lines.append("")
    return "\n".join(lines)

def _print_summary_table(results: list):
    print(f"\n{'='*90}")
    print("  COMBINED RESULTS")
    print(f"{'='*90}")

    headers = [
        "Dataset", "Raw Cands", "Cands", "T.Rem", "H.Rem", "Final",
        "Fit_base", "Prec_base", "Prec_final",
    ]
    rows = []
    for r in results:
        rows.append([
            r["dataset"],
            str(r.get("raw_candidates", "-")),
            str(r["candidates"]),
            str(r["removed_transitive"]),
            str(r["removed_heuristic"]),
            str(r["final_channels"]),
            _fmt(r["fit_base"]),
            _fmt(r["prec_base"]),
            _fmt(r["prec_final"]),
        ])

    widths = [
        max(len(h), *(len(row[i]) for row in rows))
        for i, h in enumerate(headers)
    ]
    print("  " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)))
    print("  " + " | ".join("-" * w for w in widths))
    for row in rows:
        print("  " + " | ".join(v.ljust(w) for v, w in zip(row, widths)))
    print(f"{'='*90}\n")

def _pipeline_worker(dataset, is_verbose, is_quiet, q):
    try:
        res = run_pipeline(dataset, verbose=is_verbose, quiet=is_quiet)
        q.put(("OK", res))
    except Exception as e:
        q.put(("ERROR", e))

def main():
    parser = argparse.ArgumentParser(
        description="8-Step Channel Optimization Pipeline",
    )
    parser.add_argument(
        "--dataset", type=str, default=None,
        help="Dataset name (e.g. IP-1). If omitted, runs all.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run pipeline on all datasets.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print detailed per-channel output.",
    )
    parser.add_argument(
        "--report", type=str, default=None, metavar="FILE",
        help="Save results to a markdown report file (e.g. report.md).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress step-by-step output and only show final results.",
    )
    args = parser.parse_args()

    if not DATA_DIR.exists():
        print(f"[ERROR] Data directory '{DATA_DIR}' not found.")
        sys.exit(1)

    if args.dataset and not args.all:
        datasets = [args.dataset]
    else:
        datasets = sorted(
            [d.name for d in DATA_DIR.iterdir()
             if d.is_dir() and not d.name.startswith(".")],
            key=lambda x: (
                (0, int(x.split("-")[1]))
                if "-" in x and x.split("-")[1].isdigit()
                else (1, x)
            ),
        )

    if not datasets:
        print("[ERROR] No datasets found.")
        sys.exit(1)

    batch_mode = len(datasets) > 1
    quiet_mode = batch_mode or args.quiet

    if batch_mode:
        print(f"Datasets to process: {datasets}\n")

    import multiprocessing

    all_results = []
    for idx, ds in enumerate(datasets, 1):
        if quiet_mode:
            if batch_mode:
                print(f"[{idx}/{len(datasets)}] Processing {ds}...", end=" ", flush=True)
            else:
                print(f"Processing {ds}...", end=" ", flush=True)
        
        queue = multiprocessing.Queue()
        p = multiprocessing.Process(target=_pipeline_worker, args=(ds, args.verbose, quiet_mode, queue))
        p.start()
        p.join(timeout=3600)
        
        if p.is_alive():
            if quiet_mode:
                print("TIMEOUT (1 hour) - skipped")
            else:
                print(f"[TIMEOUT] {ds} took longer than 1 hour, skipping.")
            p.terminate()
            p.join()
            continue
            
        if not queue.empty():
            status, payload = queue.get()
            if status == "ERROR":
                if quiet_mode:
                    print(f"ERROR: {payload}")
                else:
                    print(f"[ERROR] {ds}: {payload}")
                continue
            
            result = payload
            if result:
                all_results.append(result)
                if quiet_mode:
                    print(f"done  (fit={_fmt(result['fit_final'])}, "
                          f"prec={_fmt(result['prec_final'])}, "
                          f"channels={result['final_channels']})")
            else:
                if quiet_mode:
                    print("skipped")
        else:
            if quiet_mode:
                print("ERROR: Process crashed without returning a result.")
            else:
                print(f"[ERROR] {ds}: Process crashed.")

    if len(all_results) > 1:
        _print_summary_table(all_results)

    if args.report and all_results:
        if len(all_results) == 1:
            md = _build_single_report(all_results[0])
        else:
            md = _build_combined_report(all_results)

        report_path = Path(args.report)
        report_path.write_text(md, encoding="utf-8")
        print(f"Report saved → {report_path}")

if __name__ == "__main__":
    main()
