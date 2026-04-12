
from typing import Dict, List, Set, Tuple

from pm4py.objects.log.obj import EventLog

Relation = str
Channel = Tuple[str, str, Relation]

def build_precedence_map(
    log: EventLog,
    agent_attr: str,
    agent: str,
) -> Dict[Tuple[str, str], bool]:

    traces: List[List[str]] = []
    for trace in log:
        acts = [
            str(e.get("concept:name", ""))
            for e in trace
            if str(e.get(agent_attr, "")) == agent
        ]
        if acts:
            traces.append(acts)

    activities: Set[str] = set()
    for t in traces:
        activities.update(t)

    precedence: Dict[Tuple[str, str], bool] = {}
    for x in activities:
        for y in activities:
            if x == y:
                continue
            y_seen = False
            always_before = True
            for acts in traces:
                if y not in acts:
                    continue
                y_seen = True
                if x not in acts:
                    always_before = False
                    break
                first_y = acts.index(y)
                if not any(acts[i] == x for i in range(first_y)):
                    always_before = False
                    break
            precedence[(x, y)] = y_seen and always_before
    return precedence

def find_transitive_redundant(
    channels: List[Channel],
    agent1: str,
    agent2: str,
    prec_map1: Dict[Tuple[str, str], bool],
    prec_map2: Dict[Tuple[str, str], bool],
) -> List[Tuple[Channel, Channel]]:
    redundant: List[Tuple[Channel, Channel]] = []
    alive = list(channels)

    changed = True
    while changed:
        changed = False
        for cand in list(alive):
            a_act, b_act, rel = cand
            for other in alive:
                if other is cand:
                    continue
                o_a, o_b, o_rel = other
                if rel != o_rel:
                    continue

                if rel == "<":
                    if b_act != o_b:
                        continue
                    if prec_map1.get((a_act, o_a), False):
                        redundant.append((cand, other))
                        alive.remove(cand)
                        changed = True
                        break
                elif rel == ">":
                    if a_act != o_a:
                        continue
                    if prec_map2.get((b_act, o_b), False):
                        redundant.append((cand, other))
                        alive.remove(cand)
                        changed = True
                        break
            else:
                continue
            break

    return redundant

def remove_transitive_channels(
    channels: List[Channel],
    agent1: str,
    agent2: str,
    prec_map1: Dict[Tuple[str, str], bool],
    prec_map2: Dict[Tuple[str, str], bool],
) -> Tuple[List[Channel], List[Tuple[Channel, Channel]]]:
    removed_pairs = find_transitive_redundant(
        channels, agent1, agent2, prec_map1, prec_map2,
    )
    removed_set = {id(r) for r, _ in removed_pairs}
    removed_values = {r for r, _ in removed_pairs}
    kept = [ch for ch in channels if ch not in removed_values]
    return kept, removed_pairs
