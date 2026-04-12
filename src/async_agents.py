from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import pm4py
from pm4py.objects.log.obj import EventLog, Trace, Event
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils
from pm4py.convert import convert_to_event_log
import numpy as np

from utils.hybrid_metrics import compute_hybrid_metrics

Relation = str

def compute_channel_structure(
    log: EventLog,
    agent_attr: str = "org:resource",
    agents: Optional[Tuple[str, str]] = None,
) -> Tuple[
    Tuple[str, str],
    List[Tuple[str, str, str]],
    Tuple[PetriNet, Marking, Marking],
    Tuple[PetriNet, Marking, Marking],
]:
    activities: List[str] = sorted(list(_collect_activities(log)))
    rel_matrix, act_index = _build_relation_matrix_np(log, activities)

    if agents:
        agent_a_acts = _collect_agent_activities(log, agent_attr, agents[0])
        agent_b_acts = _collect_agent_activities(log, agent_attr, agents[1])
    else:
        agent_counts = _agent_frequencies(log, agent_attr)
        top_two = [a for a, _ in sorted(agent_counts.items(), key=lambda x: -x[1])[:2]]
        if len(top_two) < 2:
            raise ValueError("Log must contain at least two distinct agents")
        agent_a_acts = _collect_agent_activities(log, agent_attr, top_two[0])
        agent_b_acts = _collect_agent_activities(log, agent_attr, top_two[1])
        agents = (top_two[0], top_two[1])

    submatrix, row_labels, col_labels = _extract_submatrix_np(
        rel_matrix, act_index, agent_a_acts, agent_b_acts
    )
    minima = _find_region_minima_np(submatrix, row_labels, col_labels)
    minima = _remove_transitively_redundant(minima, None)

    log_a = _project_log_by_agent(log, agent_attr, agents[0])
    log_b = _project_log_by_agent(log, agent_attr, agents[1])
    pn_a, im_a, fm_a = _mine_perfect_fit_petri(log_a)
    pn_b, im_b, fm_b = _mine_perfect_fit_petri(log_b)

    return agents, minima, (pn_a, im_a, fm_a), (pn_b, im_b, fm_b)

def process_xes_file(
    xes_path: Path,
    agent_attr: str = "org:resource",
    agents: Optional[Tuple[str, str]] = None,
    output_dir: Path = Path("output"),
    save_svg: bool = False,
    pattern: Optional[int] = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    log_or_df = pm4py.read_xes(str(xes_path))
    if not isinstance(log_or_df, EventLog):
        log = convert_to_event_log(log_or_df)
    else:
        log = log_or_df

    agents, minima, (pn_a, im_a, fm_a), (pn_b, im_b, fm_b) = compute_channel_structure(
        log, agent_attr, agents
    )

    final_pn, final_im, final_fm = _compose_with_channels(
        pn_a, im_a, fm_a, pn_b, im_b, fm_b, minima
    )

    pnml_path = output_dir / f"final_net_{xes_path.stem}.pnml"
    pm4py.write_pnml(final_pn, final_im, final_fm, str(pnml_path))

    if pattern is not None and 1 <= pattern <= 7:
        try:
            hybrid_results = compute_hybrid_metrics(
                log=log,
                channels=minima,
                agent1=agents[0],
                agent2=agents[1],
                pattern=pattern,
                real_nets={"a1": (pn_a, im_a, fm_a), "a2": (pn_b, im_b, fm_b)},
                agent_attr=agent_attr,
            )
            print(f"Hybrid Precision(R1~A2): {hybrid_results['prec_R1_A2']:.4f}")
            print(f"Hybrid Fitness(R1~A2):   {hybrid_results['fit_R1_A2']:.4f}")
            print(f"Hybrid Precision(A1~R2): {hybrid_results['prec_A1_R2']:.4f}")
            print(f"Hybrid Fitness(A1~R2):   {hybrid_results['fit_A1_R2']:.4f}")
        except Exception as exc:
            print(f"Warning: hybrid metrics failed: {exc}")
    elif pattern is not None:
        print(f"Warning: pattern {pattern} is outside 1-7, skipping hybrid metrics.")

    if save_svg:
        svg_path = output_dir / f"final_net_{xes_path.stem}.svg"
        pm4py.save_vis_petri_net(final_pn, final_im, final_fm, str(svg_path))

def _collect_activities(log: EventLog) -> Set[str]:
    acts: Set[str] = set()
    for trace in log:
        for event in trace:
            if "concept:name" in event:
                acts.add(str(event["concept:name"]))
    return acts

def _agent_frequencies(log: EventLog, agent_attr: str) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for trace in log:
        for event in trace:
            ag = event.get(agent_attr)
            if ag is not None:
                freq[str(ag)] = freq.get(str(ag), 0) + 1
    return freq

def _collect_agent_activities(log: EventLog, agent_attr: str, agent_value: str) -> List[str]:
    acts: Set[str] = set()
    for trace in log:
        for event in trace:
            if str(event.get(agent_attr)) == agent_value:
                acts.add(str(event.get("concept:name")))
    return sorted(list(acts))

def _build_relation_matrix_np(log: EventLog, activities: List[str]) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(activities)
    act_index = {act: i for i, act in enumerate(activities)}
    matrix = np.zeros((n, n), dtype=np.int8)
    for i, a in enumerate(activities):
        for j, b in enumerate(activities):
            if i == j:
                continue
            rel = _relation_over_log(log, a, b)
            if rel == '<':
                matrix[i, j] = -1
            elif rel == '>':
                matrix[i, j] = 1
            else:
                matrix[i, j] = 0
    return matrix, act_index

def _relation_over_log(log: EventLog, a: str, b: str) -> Relation:
    always_a_before_b = True
    always_b_before_a = True
    seen_parallel = False

    for trace in log:
        positions_a = [i for i, e in enumerate(trace) if e.get("concept:name") == a]
        positions_b = [i for i, e in enumerate(trace) if e.get("concept:name") == b]
        if not positions_a or not positions_b:
            continue
        min_a = min(positions_a)
        max_a = max(positions_a)
        min_b = min(positions_b)
        max_b = max(positions_b)
        if max_a < min_b:
            always_b_before_a = False
        elif max_b < min_a:
            always_a_before_b = False
        else:
            seen_parallel = True
            always_a_before_b = False
            always_b_before_a = False

    if seen_parallel:
        return "0"
    if always_a_before_b and not always_b_before_a:
        return "<"
    if always_b_before_a and not always_a_before_b:
        return ">"
    return "0"

def _extract_submatrix_np(
    rel_matrix: np.ndarray,
    act_index: Dict[str, int],
    rows: List[str],
    cols: List[str],
) -> Tuple[np.ndarray, List[str], List[str]]:
    row_idx = [act_index[r] for r in rows if r in act_index]
    col_idx = [act_index[c] for c in cols if c in act_index]
    sub = rel_matrix[np.ix_(row_idx, col_idx)].copy()
    return sub, [rows[i] for i in range(len(row_idx))], [cols[j] for j in range(len(col_idx))]

def _find_region_minima_np(sub: np.ndarray, row_labels: List[str], col_labels: List[str]) -> List[Tuple[str, str, Relation]]:
    minima: List[Tuple[str, str, Relation]] = []
    for i in range(sub.shape[0]):
        row = sub[i, :]
        j = 0
        while j < row.shape[0]:
            if row[j] == -1:
                start = j
                while j < row.shape[0] and row[j] == -1:
                    j += 1
                minima.append((row_labels[i], col_labels[start], '<'))
            else:
                j += 1
        j = 0
        while j < row.shape[0]:
            if row[j] == 1:
                start = j
                while j < row.shape[0] and row[j] == 1:
                    j += 1
                end = j - 1
                minima.append((row_labels[i], col_labels[end], '>'))
            else:
                j += 1
    minima = list({(a, b, r) for a, b, r in minima})
    return minima

def _min_of_run(run: List[Tuple[str, str, Relation]]) -> Tuple[str, str, Relation]:
    rel = run[0][2]
    if rel == "<":
        return (run[0][0], run[0][1], "<")
    else:
        return (run[-1][0], run[-1][1], ">")

def _remove_transitively_redundant(
    minima: List[Tuple[str, str, Relation]],
    submatrix: Dict[Tuple[str, str], Relation],
) -> List[Tuple[str, str, Relation]]:
    pruned: List[Tuple[str, str, Relation]] = []
    for m in minima:
        a, b, r = m
        redundant = False
        for n in minima:
            if n == m:
                continue
            a2, b2, r2 = n
            if r == r2 and b == b2 and r == "<":
                pass
        if not redundant:
            pruned.append(m)
    uniq = list({(a, b, r) for a, b, r in pruned})
    return uniq

def _project_log_by_agent(log: EventLog, agent_attr: str, agent_value: str) -> EventLog:
    new_log = EventLog()
    for trace in log:
        new_trace = Trace()
        for event in trace:
            if str(event.get(agent_attr)) == agent_value:
                new_trace.append(event)
        if len(new_trace) > 0:
            new_log.append(new_trace)
    return new_log

def _mine_perfect_fit_petri(log: EventLog) -> Tuple[PetriNet, Marking, Marking]:
    pn, im, fm = pm4py.discover_petri_net_inductive(log)
    return pn, im, fm

def _compose_with_channels(
    pn_a: PetriNet,
    im_a: Marking,
    fm_a: Marking,
    pn_b: PetriNet,
    im_b: Marking,
    fm_b: Marking,
    minima: List[Tuple[str, str, Relation]],
) -> Tuple[PetriNet, Marking, Marking]:
    final_pn = PetriNet("multiagent_async")
    final_im = Marking()
    final_fm = Marking()

    map_places = {}
    map_trans = {}

    def copy_net(src_pn: PetriNet, src_im: Marking, src_fm: Marking, prefix: str):
        for p in src_pn.places:
            new_p = PetriNet.Place(f"{prefix}_{p.name}")
            final_pn.places.add(new_p)
            map_places[p] = new_p
        for t in src_pn.transitions:
            new_t = PetriNet.Transition(f"{prefix}_{t.name}", t.label)
            final_pn.transitions.add(new_t)
            map_trans[t] = new_t
        for a in src_pn.arcs:
            src = map_places.get(a.source) or map_trans.get(a.source)
            dst = map_places.get(a.target) or map_trans.get(a.target)
            petri_utils.add_arc_from_to(src, dst, final_pn)
        for p, c in src_im.items():
            final_im[map_places[p]] = c
        for p, c in src_fm.items():
            final_fm[map_places[p]] = c

    copy_net(pn_a, im_a, fm_a, "A")
    copy_net(pn_b, im_b, fm_b, "B")

    def find_trans_by_label(label: str, prefix: str) -> List[PetriNet.Transition]:
        return [
            t for t in final_pn.transitions
            if t.label == label and t.name.startswith(prefix + "_")
        ]

    for a_label, b_label, rel in minima:
        if rel == "<":
            place = PetriNet.Place(f"chan_{a_label}_to_{b_label}")
            final_pn.places.add(place)
            for ta in find_trans_by_label(a_label, "A"):
                petri_utils.add_arc_from_to(ta, place, final_pn)
            for tb in find_trans_by_label(b_label, "B"):
                petri_utils.add_arc_from_to(place, tb, final_pn)

        elif rel == ">":
            place = PetriNet.Place(f"chan_{b_label}_to_{a_label}")
            final_pn.places.add(place)
            for tb in find_trans_by_label(b_label, "B"):
                petri_utils.add_arc_from_to(tb, place, final_pn)
            for ta in find_trans_by_label(a_label, "A"):
                petri_utils.add_arc_from_to(place, ta, final_pn)

    return final_pn, final_im, final_fm
