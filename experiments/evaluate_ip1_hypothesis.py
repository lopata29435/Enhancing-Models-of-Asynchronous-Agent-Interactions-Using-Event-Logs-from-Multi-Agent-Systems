import copy
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import pandas as pd
import pm4py
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.conversion.log import converter as log_converter
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

def load_log_as_df(path: str) -> pd.DataFrame:
    log = pm4py.read_xes(path)
    if not isinstance(log, pd.DataFrame):
        log = log_converter.apply(log, variant=log_converter.Variants.TO_DATA_FRAME)
    if 'case:concept:name' not in log.columns and 'case:name' in log.columns:
        log = log.rename(columns={'case:name': 'case:concept:name'})
    log = log.sort_values(['case:concept:name', 'time:timestamp']).reset_index(drop=True)
    return log

def split_log_by_agent(log: pd.DataFrame) -> dict:
    return {
        agent: log[log['org:resource'] == agent].copy()
        for agent in log['org:resource'].dropna().unique()
    }

def project_hybrid_log(
    log: pd.DataFrame,
    real_agent: str,
    abstract_agent: str,
    interface_labels: set,
) -> pd.DataFrame:
    keep_real = log['org:resource'] == real_agent
    keep_abstract = (log['org:resource'] == abstract_agent) & (
        log['concept:name'].isin(interface_labels)
    )
    projected = log[keep_real | keep_abstract].copy()
    return projected.sort_values(
        ['case:concept:name', 'time:timestamp']
    ).reset_index(drop=True)

def discover_agent_net(agent_log: pd.DataFrame):
    net, im, fm = pm4py.discover_petri_net_inductive(agent_log)
    if fm is None or len(fm) == 0:
        sinks = [p for p in net.places if not p.out_arcs]
        fm = Marking({p: 1 for p in sinks}) if sinks else Marking()
    return net, im, fm

def _interface_labels(net: PetriNet, symbol: str) -> list:
    return [t for t in net.transitions if t.label and symbol in t.label]

def _counterpart_label(label: str, src_symbol: str, dst_symbol: str) -> str:
    head, _, tail = label.partition(src_symbol)
    return f"{head}{dst_symbol}{tail}"

def build_hybrid_net(
    real_net: PetriNet,
    real_im: Marking,
    real_fm: Marking,
    role: str,
    net_name: str = "hybrid",
):
    net = copy.deepcopy(real_net)
    im_places = [p for p in net.places if any(q.name == p.name for q in real_im)]
    fm_places = [p for p in net.places if any(q.name == p.name for q in real_fm)]

    net.name = net_name

    new_im_place = PetriNet.Place("new_im")
    new_fm_place = PetriNet.Place("new_fm")
    net.places.add(new_im_place)
    net.places.add(new_fm_place)

    tau_start = PetriNet.Transition("tau_start", None)
    tau_end = PetriNet.Transition("tau_end", None)
    net.transitions.add(tau_start)
    net.transitions.add(tau_end)

    petri_utils.add_arc_from_to(new_im_place, tau_start, net)
    petri_utils.add_arc_from_to(tau_end, new_fm_place, net)

    for p in im_places:
        petri_utils.add_arc_from_to(tau_start, p, net)
    for p in fm_places:
        petri_utils.add_arc_from_to(p, tau_end, net)

    p_abstract = PetriNet.Place("p_abstract")
    net.places.add(p_abstract)
    petri_utils.add_arc_from_to(tau_start, p_abstract, net)
    petri_utils.add_arc_from_to(p_abstract, tau_end, net)

    kept_labels: set = set()

    if role == "sender":
        senders = _interface_labels(net, "!")
        for snd in senders:
            recv_label = _counterpart_label(snd.label, "!", "?")
            kept_labels.add(recv_label)

            abs_recv = PetriNet.Transition(f"abs_{recv_label}", recv_label)
            net.transitions.add(abs_recv)

            channel = PetriNet.Place(f"chan_{snd.label}__{recv_label}")
            net.places.add(channel)

            petri_utils.add_arc_from_to(snd, channel, net)
            petri_utils.add_arc_from_to(channel, abs_recv, net)

            petri_utils.add_arc_from_to(p_abstract, abs_recv, net)
            petri_utils.add_arc_from_to(abs_recv, p_abstract, net)

    elif role == "receiver":
        receivers = _interface_labels(net, "?")
        for rcv in receivers:
            send_label = _counterpart_label(rcv.label, "?", "!")
            kept_labels.add(send_label)

            abs_send = PetriNet.Transition(f"abs_{send_label}", send_label)
            net.transitions.add(abs_send)

            channel = PetriNet.Place(f"chan_{send_label}__{rcv.label}")
            net.places.add(channel)

            petri_utils.add_arc_from_to(abs_send, channel, net)
            petri_utils.add_arc_from_to(channel, rcv, net)

            petri_utils.add_arc_from_to(p_abstract, abs_send, net)
            petri_utils.add_arc_from_to(abs_send, p_abstract, net)
    else:
        raise ValueError("role must be 'sender' or 'receiver'")

    new_im = Marking({new_im_place: 1})
    new_fm = Marking({new_fm_place: 1})
    return net, new_im, new_fm, kept_labels

def compute_precision(log_df: pd.DataFrame, net: PetriNet, im: Marking, fm: Marking) -> float:
    event_log = log_converter.apply(
        log_df, variant=log_converter.Variants.TO_EVENT_LOG
    )

    try:
        variant = precision_evaluator.Variants.ALIGN_ETCONFORMANCE
    except AttributeError:
        variant = precision_evaluator.Variants.ALIGN_ETCP

    return precision_evaluator.apply(event_log, net, im, fm, variant=variant)

def evaluate_ip1(
    log_path: str = "data/IP-1/IP-1_init_log.xes",
    sender_agent: str = "Agent 1",
    receiver_agent: str = "Agent 2",
    out_dir: str = "output/IP-1/hybrid",
):
    os.makedirs(out_dir, exist_ok=True)

    log = load_log_as_df(log_path)
    per_agent = split_log_by_agent(log)

    if sender_agent not in per_agent or receiver_agent not in per_agent:
        raise RuntimeError(
            f"Expected agents {sender_agent!r} and {receiver_agent!r}, "
            f"found {list(per_agent)}"
        )

    print(f"[1/4] Discovering detailed nets for {sender_agent} and {receiver_agent}")
    r1_net, r1_im, r1_fm = discover_agent_net(per_agent[sender_agent])
    r2_net, r2_im, r2_fm = discover_agent_net(per_agent[receiver_agent])

    print("[2/4] Building hybrid model R1 ~ A2 (sender + abstract receiver)")
    h1_net, h1_im, h1_fm, recv_labels = build_hybrid_net(
        r1_net, r1_im, r1_fm, role="sender", net_name="R1_tilde_A2"
    )
    h1_log = project_hybrid_log(log, sender_agent, receiver_agent, recv_labels)

    pnml_r1 = os.path.join(out_dir, "IP-1_hybrid_R1_A2.pnml")
    xes_r1 = os.path.join(out_dir, "IP-1_hybrid_R1_A2_log.xes")
    pm4py.write_pnml(h1_net, h1_im, h1_fm, pnml_r1)
    pm4py.write_xes(
        log_converter.apply(h1_log, variant=log_converter.Variants.TO_EVENT_LOG),
        xes_r1,
    )

    print("[3/4] Building hybrid model A1 ~ R2 (abstract sender + receiver)")
    h2_net, h2_im, h2_fm, send_labels = build_hybrid_net(
        r2_net, r2_im, r2_fm, role="receiver", net_name="A1_tilde_R2"
    )
    h2_log = project_hybrid_log(log, receiver_agent, sender_agent, send_labels)

    pnml_r2 = os.path.join(out_dir, "IP-1_hybrid_A1_R2.pnml")
    xes_r2 = os.path.join(out_dir, "IP-1_hybrid_A1_R2_log.xes")
    pm4py.write_pnml(h2_net, h2_im, h2_fm, pnml_r2)
    pm4py.write_xes(
        log_converter.apply(h2_log, variant=log_converter.Variants.TO_EVENT_LOG),
        xes_r2,
    )

    print("[4/4] Computing precision on each hybrid <log, model> pair")
    prec_r1_a2 = compute_precision(h1_log, h1_net, h1_im, h1_fm)
    prec_a1_r2 = compute_precision(h2_log, h2_net, h2_im, h2_fm)

    min_prec = min(prec_r1_a2, prec_a1_r2)

    print("\n" + "=" * 60)
    print("IP-1 HYBRID PRECISION REPORT")
    print("=" * 60)
    print(f"Interface labels kept for R1 ~ A2 : {sorted(recv_labels)}")
    print(f"Interface labels kept for A1 ~ R2 : {sorted(send_labels)}")
    print(f"Hybrid log size (R1 ~ A2)         : {len(h1_log)} events / "
          f"{h1_log['case:concept:name'].nunique()} traces")
    print(f"Hybrid log size (A1 ~ R2)         : {len(h2_log)} events / "
          f"{h2_log['case:concept:name'].nunique()} traces")
    print("-" * 60)
    print(f"Precision( R1 ~ A2 ) = {prec_r1_a2:.6f}")
    print(f"Precision( A1 ~ R2 ) = {prec_a1_r2:.6f}")
    print(f"Minimum single-agent precision = {min_prec:.6f}")
    print("=" * 60)
    print(f"Hybrid artefacts saved under: {out_dir}")

    return {
        "prec_R1_A2": prec_r1_a2,
        "prec_A1_R2": prec_a1_r2,
        "min_precision": min_prec,
        "hybrid_model_R1_A2": pnml_r1,
        "hybrid_model_A1_R2": pnml_r2,
        "hybrid_log_R1_A2": xes_r1,
        "hybrid_log_A1_R2": xes_r2,
    }

def main():
    evaluate_ip1()

if __name__ == "__main__":
    main()
