import glob
import os
import sys
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(HERE, '..', 'src'))
sys.path.append(HERE)

import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.petri_net.exporter import exporter as pnml_exporter

from compute_fitness import generate_final_marking_from_net
from log_parser import XESLogParser
from synthesis_algorithm import MultiAgentPetriNetSynthesizer

from evaluate_ip1_hypothesis import (
    build_hybrid_net,
    compute_precision,
    load_log_as_df,
    project_hybrid_log,
)

SUPPORTED_DATASET = "IP-1"
SENDER_AGENT = "Agent 1"
RECEIVER_AGENT = "Agent 2"

def eval_net(log, net, im, fm=None):
    if fm is None:
        fm = generate_final_marking_from_net(net)

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

    tr_res = token_replay.apply(log, net, im, fm)
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
            trace_fitnesses.append(float(r))

    fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0

    try:
        try:
            var = precision_evaluator.Variants.ETCONFORMANCE_TOKEN
        except AttributeError:
            var = precision_evaluator.Variants.ALIGN_ETCP
        precision = precision_evaluator.apply(log, net, im, fm, variant=var)
    except Exception as e:
        print(f"    * Warning: Could not compute precision: {e}")
        precision = 0.0

    return fitness, precision

def evaluate_hybrid_agent_precisions(
    full_log_df,
    r1_net, r1_im, r1_fm,
    r2_net, r2_im, r2_fm,
    sender_agent=SENDER_AGENT,
    receiver_agent=RECEIVER_AGENT,
    hybrid_out_dir=None,
):
    h1_net, h1_im, h1_fm, recv_labels = build_hybrid_net(
        r1_net, r1_im, r1_fm, role="sender", net_name="R1_tilde_A2"
    )
    h1_log = project_hybrid_log(
        full_log_df, sender_agent, receiver_agent, recv_labels
    )

    h2_net, h2_im, h2_fm, send_labels = build_hybrid_net(
        r2_net, r2_im, r2_fm, role="receiver", net_name="A1_tilde_R2"
    )
    h2_log = project_hybrid_log(
        full_log_df, receiver_agent, sender_agent, send_labels
    )

    if hybrid_out_dir:
        os.makedirs(hybrid_out_dir, exist_ok=True)
        from pm4py.objects.conversion.log import converter as log_converter

        pm4py.write_pnml(
            h1_net, h1_im, h1_fm,
            os.path.join(hybrid_out_dir, "IP-1_hybrid_R1_A2.pnml"),
        )
        pm4py.write_xes(
            log_converter.apply(
                h1_log, variant=log_converter.Variants.TO_EVENT_LOG
            ),
            os.path.join(hybrid_out_dir, "IP-1_hybrid_R1_A2_log.xes"),
        )
        pm4py.write_pnml(
            h2_net, h2_im, h2_fm,
            os.path.join(hybrid_out_dir, "IP-1_hybrid_A1_R2.pnml"),
        )
        pm4py.write_xes(
            log_converter.apply(
                h2_log, variant=log_converter.Variants.TO_EVENT_LOG
            ),
            os.path.join(hybrid_out_dir, "IP-1_hybrid_A1_R2_log.xes"),
        )

    prec_r1_a2 = compute_precision(h1_log, h1_net, h1_im, h1_fm)
    prec_a1_r2 = compute_precision(h2_log, h2_net, h2_im, h2_fm)
    return prec_r1_a2, prec_a1_r2

def build_precedence_matrix(sub_log) -> dict:
    activities = set()
    for trace in sub_log:
        for event in trace:
            activities.add(event["concept:name"])

    traces_as_lists = [
        [e["concept:name"] for e in trace] for trace in sub_log
    ]

    precedence = {}
    for x in activities:
        for y in activities:
            if x == y:
                continue
            y_was_seen = False
            always_before = True
            for acts in traces_as_lists:
                if y not in acts:
                    continue
                y_was_seen = True
                if x not in acts:
                    always_before = False
                    break
                first_y = acts.index(y)
                if not any(
                    acts[i] == x for i in range(first_y)
                ):
                    always_before = False
                    break
            precedence[(x, y)] = y_was_seen and always_before
    return precedence

def _channel_key(ch):
    return (ch[0], ch[1], ch[2], ch[3])

def is_transitively_redundant(channel, other_channels, precedence_per_agent):
    src_agent, tgt_agent, src_act, tgt_act = _channel_key(channel)
    src_prec = precedence_per_agent.get(src_agent, {})
    tgt_prec = precedence_per_agent.get(tgt_agent, {})

    for other in other_channels:
        if _channel_key(other) == _channel_key(channel):
            continue
        o_src_agent, o_tgt_agent, o_src_act, o_tgt_act = _channel_key(other)
        if o_src_agent != src_agent or o_tgt_agent != tgt_agent:
            continue
        src_ok = (src_act == o_src_act) or src_prec.get((src_act, o_src_act), False)
        tgt_ok = (o_tgt_act == tgt_act) or tgt_prec.get((o_tgt_act, tgt_act), False)
        if src_ok and tgt_ok:
            return True, other
    return False, None

def remove_channel_from_net(sys_net, channel_name: str) -> bool:
    target_place = None
    for p in list(sys_net.places):
        if p.name == channel_name:
            target_place = p
            break
    if target_place is None:
        return False

    for arc in list(target_place.in_arcs):
        sys_net.arcs.discard(arc)
        try:
            arc.source.out_arcs.discard(arc)
        except Exception:
            pass
    for arc in list(target_place.out_arcs):
        sys_net.arcs.discard(arc)
        try:
            arc.target.in_arcs.discard(arc)
        except Exception:
            pass
    sys_net.places.discard(target_place)
    return True

def channel_place_name(src_agent, src_act, tgt_agent, tgt_act) -> str:
    return f"chan_{src_agent}_{src_act}_to_{tgt_agent}_{tgt_act}"

def structural_optimization(synthesizer, sys_net):
    channels = []
    for src, tgt, s_act, t_act in synthesizer.acyclic_channels:
        channels.append((src, tgt, s_act, t_act, 0, "acyclic"))
    for src, tgt, s_act, t_act, k in synthesizer.cyclic_channels:
        channels.append((src, tgt, s_act, t_act, k, "cyclic"))

    precedence_per_agent = {
        agent: build_precedence_matrix(sub_log)
        for agent, sub_log in synthesizer.sub_logs.items()
    }

    alive = list(channels)
    removed = []
    witnesses = {}

    print(f"    -> Structural analysis starts with {len(alive)} candidate channels.")

    while True:
        victim = None
        witness = None
        for cand in alive:
            redundant, w = is_transitively_redundant(
                cand, alive, precedence_per_agent
            )
            if redundant:
                victim = cand
                witness = w
                break
        if victim is None:
            break

        place_name = channel_place_name(victim[0], victim[2], victim[1], victim[3])
        removed_ok = remove_channel_from_net(sys_net, place_name)
        if not removed_ok:
            print(f"    -> [warn] channel place not found: {place_name}")
        alive.remove(victim)
        removed.append(victim)
        witnesses[_channel_key(victim)] = _channel_key(witness)
        print(
            f"    -> REMOVED (transitively redundant): "
            f"{victim[0]}({victim[2]}) -> {victim[1]}({victim[3]})  "
            f"| implied by {witness[0]}({witness[2]}) -> "
            f"{witness[1]}({witness[3]})"
        )

    for ch in alive:
        print(
            f"    -> KEPT: "
            f"{ch[0]}({ch[2]}) -> {ch[1]}({ch[3]})"
        )

    return removed, alive, witnesses

def _locate_log(data_dir, ds):
    data_path = os.path.join(data_dir, ds)
    candidates = glob.glob(os.path.join(data_path, "*_init_log_enriched.xes"))
    if not candidates:
        candidates = glob.glob(os.path.join(data_path, "*.xes"))
    candidates = [
        f for f in candidates
        if "__MACOSX" not in f and not os.path.basename(f).startswith('.')
    ]
    return candidates[0] if candidates else None

def process_dataset(ds, data_dir="data", output_dir="output"):
    if ds != SUPPORTED_DATASET:
        raise ValueError(
            f"This runner only supports '{SUPPORTED_DATASET}', got '{ds}'"
        )

    print(f"\n{'=' * 60}")
    print(f"IP-1 PIPELINE - hybrid baseline + transitivity optimization")
    print(f"{'=' * 60}")

    log_file = _locate_log(data_dir, ds)
    if not log_file:
        print(f"[ERROR] No valid log found for {ds}.")
        return None

    print(f"[*] Log file: {log_file}")
    try:
        log = XESLogParser.load_log(log_file)
    except Exception as e:
        print(f"[ERROR] Error loading log: {e}")
        return None

    full_log_df = load_log_as_df(log_file)

    try:
        synthesizer = MultiAgentPetriNetSynthesizer(log)
        sys_net, sys_im, sys_fm = synthesizer.run(apply_transitivity=False)

        print("\n[Step 1] BASELINE MEASUREMENT - hybrid per-agent precision")
        r1_net, r1_im, r1_fm = synthesizer.local_nets[SENDER_AGENT]
        r2_net, r2_im, r2_fm = synthesizer.local_nets[RECEIVER_AGENT]

        hybrid_dir = os.path.join(output_dir, ds, "hybrid")
        prec_r1_a2, prec_a1_r2 = evaluate_hybrid_agent_precisions(
            full_log_df,
            r1_net, r1_im, r1_fm,
            r2_net, r2_im, r2_fm,
            sender_agent=SENDER_AGENT,
            receiver_agent=RECEIVER_AGENT,
            hybrid_out_dir=hybrid_dir,
        )
        min_ag_p = min(prec_r1_a2, prec_a1_r2)
        print(f"    -> Precision( R1 ~ A2 ) = {prec_r1_a2:.4f}")
        print(f"    -> Precision( A1 ~ R2 ) = {prec_a1_r2:.4f}")
        print(f"    -> min_ag_p             = {min_ag_p:.4f}")

        print("\n[Step 2] BASELINE SYSTEM SYNTHESIS - all channels present")
        baseline_f, baseline_p = eval_net(log, sys_net, sys_im, sys_fm)
        n_initial = len(
            [p for p in sys_net.places if str(p.name).startswith("chan_")]
        )
        print(f"    -> Channels present        : {n_initial}")
        print(f"    -> Baseline System Fitness : {baseline_f:.4f}")
        print(f"    -> Baseline System Prec.   : {baseline_p:.4f}")

        baseline_pnml = os.path.join(output_dir, ds, f"{ds}_composition_baseline.pnml")
        os.makedirs(os.path.dirname(baseline_pnml), exist_ok=True)
        pnml_exporter.apply(sys_net, sys_im, baseline_pnml, final_marking=sys_fm)

        print("\n[Step 3] STRUCTURAL OPTIMIZATION - transitivity-based removal")
        removed, kept, witnesses = structural_optimization(synthesizer, sys_net)

        print("\n[Step 4] FINAL CHECK - optimized system precision")
        final_f, final_p = eval_net(log, sys_net, sys_im, sys_fm)
        n_final = len(
            [p for p in sys_net.places if str(p.name).startswith("chan_")]
        )
        print(f"    -> Channels remaining      : {n_final} "
              f"(removed {len(removed)}/{n_initial})")
        print(f"    -> Optimized Fitness       : {final_f:.4f}")
        print(f"    -> Optimized Precision     : {final_p:.4f}")

        optimized_pnml = os.path.join(output_dir, ds, f"{ds}_composition_optimized.pnml")
        pnml_exporter.apply(sys_net, sys_im, optimized_pnml, final_marking=sys_fm)

        print("\n[Step 5] CONCLUSION")
        hypothesis_holds = final_p >= min_ag_p
        if hypothesis_holds:
            print(
                f"    -> HYPOTHESIS CONFIRMED: "
                f"final Precision {final_p:.4f} >= min_ag_p {min_ag_p:.4f}."
            )
        else:
            print(
                f"    -> HYPOTHESIS NOT CONFIRMED: "
                f"final Precision {final_p:.4f} < min_ag_p {min_ag_p:.4f}."
            )

        print("\n" + "-" * 60)
        print("SUMMARY")
        print("-" * 60)
        print(f"  Baseline Fitness / Precision   : {baseline_f:.4f} / {baseline_p:.4f}")
        print(f"  Optimized Fitness / Precision  : {final_f:.4f} / {final_p:.4f}")
        print(f"  Precision( R1 ~ A2 )           : {prec_r1_a2:.4f}")
        print(f"  Precision( A1 ~ R2 )           : {prec_a1_r2:.4f}")
        print(f"  min_ag_p                       : {min_ag_p:.4f}")
        print(f"  Channels: initial={n_initial}, removed={len(removed)}, kept={n_final}")
        print(f"  Hypothesis confirmed?          : {'Yes' if hypothesis_holds else 'No'}")
        print(f"  Baseline PNML  -> {baseline_pnml}")
        print(f"  Optimized PNML -> {optimized_pnml}")
        print(f"  Hybrid artefacts -> {hybrid_dir}")

        return {
            "Pattern": ds,
            "Desc": f"Pattern {ds} (5-step pipeline)",
            "Baseline_f": f"{baseline_f:.4f}",
            "Baseline_p": f"{baseline_p:.4f}",
            "Final_f": f"{final_f:.4f}",
            "Final_p": f"{final_p:.4f}",
            "Prec_R1_A2": f"{prec_r1_a2:.4f}",
            "Prec_A1_R2": f"{prec_a1_r2:.4f}",
            "Min_ag_p": f"{min_ag_p:.4f}",
            "Hyp": "Yes" if hypothesis_holds else "No",
            "Note": (
                f"Removed {len(removed)}/{n_initial} channels by transitivity"
            ),
        }

    except Exception as e:
        print(f"[ERROR] Error during the pipeline: {e}")
        traceback.print_exc()
        return None

def run_all_datasets(data_dir="data"):
    print("[INFO] This IP-1-only runner ignores other datasets.")
    res = process_dataset(SUPPORTED_DATASET, data_dir)
    if res is None:
        return

    output_md_path = "evaluation_results_table_ip1.md"
    print(f"\nWriting results to {output_md_path}...")
    with open(output_md_path, "w", encoding="utf-8") as f:
        f.write("# IP-1 Pipeline (hybrid baseline + transitivity optimization)\n\n")
        f.write(
            "| Pattern | Description | Baseline Fit | Baseline Prec | "
            "Final Fit | Final Prec | Prec(R1~A2) | Prec(A1~R2) | min_ag_p | "
            "Hypothesis | Note |\n"
        )
        f.write(
            "|---------|-------------|--------------|---------------|"
            "-----------|------------|-------------|-------------|----------|"
            "-----------|------|\n"
        )
        f.write(
            f"| {res['Pattern']} | {res['Desc']} | "
            f"{res['Baseline_f']} | {res['Baseline_p']} | "
            f"{res['Final_f']} | {res['Final_p']} | "
            f"{res['Prec_R1_A2']} | {res['Prec_A1_R2']} | "
            f"{res['Min_ag_p']} | {res['Hyp']} | {res['Note']} |\n"
        )
    print(f"\nDone! See {output_md_path}.")

if __name__ == "__main__":
    run_all_datasets()
