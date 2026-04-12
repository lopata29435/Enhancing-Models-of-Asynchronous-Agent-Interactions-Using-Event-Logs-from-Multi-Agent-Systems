
import random
from typing import Callable, List, Optional, Tuple

from pm4py.objects.petri_net.obj import PetriNet, Marking

def _find_channel_places(net: PetriNet) -> List[PetriNet.Place]:
    return [p for p in net.places if p.name.startswith("chan_")]

def remove_channel_from_net(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    place_name: str,
) -> Optional[Tuple[PetriNet.Place, list, list, int, int]]:
    target = None
    for p in net.places:
        if p.name == place_name:
            target = p
            break
    if target is None:
        return None

    in_arcs = list(target.in_arcs)
    out_arcs = list(target.out_arcs)

    for arc in in_arcs:
        arc.source.out_arcs.discard(arc)
        net.arcs.discard(arc)
    for arc in out_arcs:
        arc.target.in_arcs.discard(arc)
        net.arcs.discard(arc)
    net.places.discard(target)

    i_tok = im.pop(target, 0)
    f_tok = fm.pop(target, 0)

    return target, in_arcs, out_arcs, i_tok, f_tok

def restore_channel_to_net(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    place: PetriNet.Place,
    in_arcs: list,
    out_arcs: list,
    i_tok: int,
    f_tok: int,
) -> None:
    net.places.add(place)
    for arc in in_arcs:
        arc.source.out_arcs.add(arc)
        net.arcs.add(arc)
    for arc in out_arcs:
        arc.target.in_arcs.add(arc)
        net.arcs.add(arc)
    if i_tok > 0:
        im[place] = i_tok
    if f_tok > 0:
        fm[place] = f_tok

MetricsFn = Callable[[PetriNet, Marking, Marking], Tuple[float, float]]

def iterative_random_removal(
    net: PetriNet,
    im: Marking,
    fm: Marking,
    metrics_fn: MetricsFn,
    min_ag_p: float,
    prec_disconnected: float,
    seed: int = 42,
    quiet: bool = True,
) -> Tuple[int, int, List[str], List[str]]:
    channels = _find_channel_places(net)
    channel_names = [p.name for p in channels]

    rng = random.Random(seed)
    rng.shuffle(channel_names)

    n_initial = len(channel_names)
    removed_names: List[str] = []
    kept_names: List[str] = []

    for idx, ch_name in enumerate(channel_names, 1):
        info = remove_channel_from_net(net, im, fm, ch_name)
        if info is None:
            if not quiet:
                print(f"  [{idx}/{n_initial}] {ch_name} — not found, skip")
            continue

        place, in_arcs, out_arcs, i_tok, f_tok = info

        try:
            _, prec_current = metrics_fn(net, im, fm)
        except Exception:
            prec_current = 0.0

        if prec_current >= min_ag_p and prec_current > prec_disconnected:
            removed_names.append(ch_name)
            if not quiet:
                print(
                    f"  [{idx}/{n_initial}] REMOVED {ch_name}  "
                    f"(prec={prec_current:.4f} >= min_ag_p={min_ag_p:.4f}, "
                    f">= disc={prec_disconnected:.4f})"
                )
        else:
            restore_channel_to_net(
                net, im, fm, place, in_arcs, out_arcs, i_tok, f_tok,
            )
            kept_names.append(ch_name)
            if not quiet:
                print(
                    f"  [{idx}/{n_initial}] KEPT    {ch_name}  "
                    f"(prec={prec_current:.4f}; thresholds: "
                    f"min_ag_p={min_ag_p:.4f}, disc={prec_disconnected:.4f})"
                )

    tried = set(removed_names) | set(kept_names)
    for p in _find_channel_places(net):
        if p.name not in tried:
            kept_names.append(p.name)

    return n_initial, len(removed_names), removed_names, kept_names
