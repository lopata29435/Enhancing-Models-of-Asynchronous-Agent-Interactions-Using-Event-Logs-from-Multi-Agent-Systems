from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pm4py.objects.log.obj import EventLog, Trace
from pm4py.objects.petri_net.obj import Marking, PetriNet
from pm4py.objects.petri_net.utils import petri_utils

from src.compute_metrics import compute_metrics_from_objects

ChannelList = List[Tuple[str, str, str]]

def extract_agent_interface_labels(
    channels: ChannelList,
    side: int,
) -> Tuple[Set[str], Set[str]]:
    receives: Set[str] = set()
    sends: Set[str] = set()
    for a1_act, a2_act, rel in channels:
        if side == 1:
            if rel == '<':
                sends.add(a1_act)
            elif rel == '>':
                receives.add(a1_act)
        else:
            if rel == '<':
                receives.add(a2_act)
            elif rel == '>':
                sends.add(a2_act)
    return receives, sends

def detect_exchange_pairs(
    log: EventLog,
    agent_attr: str,
    agent: str,
    receives: Set[str],
    sends: Set[str],
) -> Dict[str, str]:
    if not receives or not sends:
        return {}

    vote: Dict[str, Dict[str, int]] = {}
    for trace in log:
        ag_events = [
            str(e.get("concept:name", ""))
            for e in trace
            if str(e.get(agent_attr, "")) == agent
        ]
        for i, name in enumerate(ag_events):
            if name in receives:
                for j in range(i + 1, len(ag_events)):
                    nxt = ag_events[j]
                    if nxt in sends:
                        vote.setdefault(name, {}).setdefault(nxt, 0)
                        vote[name][nxt] += 1
                        break

    return {recv: max(cnt, key=cnt.get) for recv, cnt in vote.items() if cnt}

def build_abstract_net(
    pattern: int,
    receives: List[str],
    sends: List[str],
    pairs: Dict[str, str],
    prefix: str = "abs",
) -> Tuple[PetriNet, Marking, Marking]:
    net = PetriNet(f"{prefix}_abstract")

    def _place(name: str) -> PetriNet.Place:
        p = PetriNet.Place(f"{prefix}_{name}")
        net.places.add(p)
        return p

    def _trans(name: str, label: Optional[str]) -> PetriNet.Transition:
        t = PetriNet.Transition(f"{prefix}_{name}", label)
        net.transitions.add(t)
        return t

    def _arc(src, dst) -> None:
        petri_utils.add_arc_from_to(src, dst, net)

    if pattern == 1:
        p0 = _place("p0")
        p1 = _place("p1")
        r = _trans("recv", receives[0] if receives else None)
        _arc(p0, r); _arc(r, p1)
        return net, Marking({p0: 1}), Marking({p1: 1})

    if pattern == 2:
        p0 = _place("p0")
        p_end = _place("p_end")
        for i, recv_label in enumerate(receives):
            ri = _trans(f"recv_{i}", recv_label)
            _arc(p0, ri)
            _arc(ri, p_end)
        return net, Marking({p0: 1}), Marking({p_end: 1})

    if pattern == 3:
        p0 = _place("p0")
        p_end = _place("p_end")
        for i, recv_label in enumerate(receives):
            ri = _trans(f"recv_{i}", recv_label)
            _arc(p0, ri)
            _arc(ri, p_end)
        return net, Marking({p0: 1}), Marking({p_end: 1})

    if pattern == 4:
        p0 = _place("p0")
        p1 = _place("p1")
        p2 = _place("p2")
        r = _trans("recv", receives[0] if receives else None)
        s = _trans("send", sends[0] if sends else None)
        _arc(p0, r); _arc(r, p1); _arc(p1, s); _arc(s, p2)
        return net, Marking({p0: 1}), Marking({p2: 1})

    if pattern == 5:
        p0 = _place("p0")
        p_end = _place("p_end")
        for i, recv_label in enumerate(receives):
            send_label = pairs.get(recv_label) or (sends[i] if i < len(sends) else None)
            p_mid = _place(f"pm_{i}")
            ri = _trans(f"recv_{i}", recv_label)
            si = _trans(f"send_{i}", send_label)
            _arc(p0, ri); _arc(ri, p_mid); _arc(p_mid, si); _arc(si, p_end)
        return net, Marking({p0: 1}), Marking({p_end: 1})

    if pattern == 6:
        p0 = _place("p0")
        p_end = _place("p_end")
        for i, recv_label in enumerate(receives):
            send_label = pairs.get(recv_label) or (sends[i] if i < len(sends) else None)
            p_mid = _place(f"pm_{i}")
            ri = _trans(f"recv_{i}", recv_label)
            si = _trans(f"send_{i}", send_label)
            _arc(p0, ri); _arc(ri, p_mid); _arc(p_mid, si); _arc(si, p_end)
        return net, Marking({p0: 1}), Marking({p_end: 1})

    if pattern == 7:
        p_loop = _place("p_loop")
        p_mid = _place("p_mid")
        p_final = _place("p_final")
        r = _trans("recv", receives[0] if receives else None)
        s = _trans("send", sends[0] if sends else None)
        tau_exit = _trans("tau_exit", None)
        _arc(p_loop, r); _arc(r, p_mid); _arc(p_mid, s); _arc(s, p_loop)
        _arc(p_loop, tau_exit); _arc(tau_exit, p_final)
        return net, Marking({p_loop: 1}), Marking({p_final: 1})

    raise ValueError(f"Unsupported pattern: {pattern}")

def _find_by_label(
    net: PetriNet, label: str, name_prefix: str
) -> List[PetriNet.Transition]:
    return [
        t for t in net.transitions
        if t.label == label and t.name.startswith(name_prefix + "_")
    ]

def compose_hybrid_net(
    real_net: PetriNet,
    real_im: Marking,
    real_fm: Marking,
    abstract_net: PetriNet,
    abstract_im: Marking,
    abstract_fm: Marking,
    channels: ChannelList,
    real_side: int,
    quiet: bool = True,
) -> Tuple[PetriNet, Marking, Marking]:
    hybrid = PetriNet("hybrid")
    r_place_map: Dict[PetriNet.Place, PetriNet.Place] = {}
    r_trans_map: Dict[PetriNet.Transition, PetriNet.Transition] = {}
    a_place_map: Dict[PetriNet.Place, PetriNet.Place] = {}
    a_trans_map: Dict[PetriNet.Transition, PetriNet.Transition] = {}

    def _copy_net(src: PetriNet, src_im: Marking, src_fm: Marking,
                  pfx: str, pmap, tmap):
        for p in src.places:
            new_p = PetriNet.Place(f"{pfx}_{p.name}")
            hybrid.places.add(new_p)
            pmap[p] = new_p
        for t in src.transitions:
            new_t = PetriNet.Transition(f"{pfx}_{t.name}", t.label)
            hybrid.transitions.add(new_t)
            tmap[t] = new_t
        for arc in src.arcs:
            src_node = pmap.get(arc.source) or tmap.get(arc.source)
            dst_node = pmap.get(arc.target) or tmap.get(arc.target)
            petri_utils.add_arc_from_to(src_node, dst_node, hybrid)

    _copy_net(real_net, real_im, real_fm, "R", r_place_map, r_trans_map)
    _copy_net(abstract_net, abstract_im, abstract_fm, "A", a_place_map, a_trans_map)

    for a1_act, a2_act, rel in channels:
        if real_side == 1:
            if rel == '<':
                sender_label, receiver_label = a1_act, a2_act
                sender_prefix, receiver_prefix = "R", "A"
            else:
                sender_label, receiver_label = a2_act, a1_act
                sender_prefix, receiver_prefix = "A", "R"
        else:
            if rel == '<':
                sender_label, receiver_label = a1_act, a2_act
                sender_prefix, receiver_prefix = "A", "R"
            else:
                sender_label, receiver_label = a2_act, a1_act
                sender_prefix, receiver_prefix = "R", "A"

        senders = _find_by_label(hybrid, sender_label, sender_prefix)
        receivers = _find_by_label(hybrid, receiver_label, receiver_prefix)

        if not senders:
            if not quiet:
                print(f"Warning: no '{sender_label}' transition found with prefix '{sender_prefix}'")
            continue
        if not receivers:
            if not quiet:
                print(f"Warning: no '{receiver_label}' transition found with prefix '{receiver_prefix}'")
            continue

        chan = PetriNet.Place(f"chan_{sender_label}__{receiver_label}")
        hybrid.places.add(chan)
        for t in senders:
            petri_utils.add_arc_from_to(t, chan, hybrid)
        for t in receivers:
            petri_utils.add_arc_from_to(chan, t, hybrid)

    p_new_im = PetriNet.Place("global_im")
    p_new_fm = PetriNet.Place("global_fm")
    hybrid.places.add(p_new_im)
    hybrid.places.add(p_new_fm)

    tau_start = PetriNet.Transition("tau_start", None)
    tau_end = PetriNet.Transition("tau_end", None)
    hybrid.transitions.add(tau_start)
    hybrid.transitions.add(tau_end)

    petri_utils.add_arc_from_to(p_new_im, tau_start, hybrid)
    petri_utils.add_arc_from_to(tau_end, p_new_fm, hybrid)

    for orig_p in real_im:
        petri_utils.add_arc_from_to(tau_start, r_place_map[orig_p], hybrid)
    for orig_p in abstract_im:
        petri_utils.add_arc_from_to(tau_start, a_place_map[orig_p], hybrid)

    for orig_p in real_fm:
        petri_utils.add_arc_from_to(r_place_map[orig_p], tau_end, hybrid)
    for orig_p in abstract_fm:
        petri_utils.add_arc_from_to(a_place_map[orig_p], tau_end, hybrid)

    return hybrid, Marking({p_new_im: 1}), Marking({p_new_fm: 1})

def project_hybrid_log(
    log: EventLog,
    agent_attr: str,
    real_agent: str,
    abstract_agent: str,
    interface_labels: Set[str],
) -> EventLog:
    new_log = EventLog()
    for trace in log:
        new_trace = Trace()
        for k, v in trace.attributes.items():
            new_trace.attributes[k] = v
        for event in trace:
            ag = str(event.get(agent_attr, ""))
            name = str(event.get("concept:name", ""))
            if ag == real_agent:
                new_trace.append(event)
            elif ag == abstract_agent and name in interface_labels:
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log

def compute_hybrid_metrics(
    log: EventLog,
    channels: ChannelList,
    agent1: str,
    agent2: str,
    pattern: int,
    real_nets: Dict[str, Tuple[PetriNet, Marking, Marking]],
    agent_attr: str = "org:resource",
) -> Dict[str, float]:
    recv1, send1 = extract_agent_interface_labels(channels, side=1)
    recv2, send2 = extract_agent_interface_labels(channels, side=2)

    pairs2 = detect_exchange_pairs(log, agent_attr, agent2, recv2, send2)
    pairs1 = detect_exchange_pairs(log, agent_attr, agent1, recv1, send1)

    abs_a2_net, abs_a2_im, abs_a2_fm = build_abstract_net(
        pattern, sorted(recv2), sorted(send2), pairs2, prefix="A2"
    )
    abs_a1_net, abs_a1_im, abs_a1_fm = build_abstract_net(
        pattern, sorted(recv1), sorted(send1), pairs1, prefix="A1"
    )

    h1_net, h1_im, h1_fm = compose_hybrid_net(
        real_nets['a1'][0], real_nets['a1'][1], real_nets['a1'][2],
        abs_a2_net, abs_a2_im, abs_a2_fm,
        channels, real_side=1,
    )
    h2_net, h2_im, h2_fm = compose_hybrid_net(
        real_nets['a2'][0], real_nets['a2'][1], real_nets['a2'][2],
        abs_a1_net, abs_a1_im, abs_a1_fm,
        channels, real_side=2,
    )

    iface2 = recv2 | send2
    iface1 = recv1 | send1
    log_h1 = project_hybrid_log(log, agent_attr, agent1, agent2, iface2)
    log_h2 = project_hybrid_log(log, agent_attr, agent2, agent1, iface1)

    import sys
    print("    Computing R1~A2 metrics...", flush=True)
    fit1, prec1, _ = compute_metrics_from_objects(log_h1, h1_net, h1_im, h1_fm, quiet=False)
    
    print("    Computing A1~R2 metrics...", flush=True)
    fit2, prec2, _ = compute_metrics_from_objects(log_h2, h2_net, h2_im, h2_fm, quiet=False)

    return {
        "prec_R1_A2": prec1,
        "fit_R1_A2": fit1,
        "prec_A1_R2": prec2,
        "fit_A1_R2": fit2,
    }
