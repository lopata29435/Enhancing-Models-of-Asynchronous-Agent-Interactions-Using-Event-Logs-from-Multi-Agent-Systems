
from typing import Dict, Tuple

from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils import petri_utils

def compose_disconnected(
    pn_a: PetriNet,
    im_a: Marking,
    fm_a: Marking,
    pn_b: PetriNet,
    im_b: Marking,
    fm_b: Marking,
) -> Tuple[PetriNet, Marking, Marking]:
    net = PetriNet("disconnected_agents")
    new_im = Marking()
    new_fm = Marking()

    place_map: Dict[PetriNet.Place, PetriNet.Place] = {}
    trans_map: Dict[PetriNet.Transition, PetriNet.Transition] = {}

    def _copy(src: PetriNet, src_im: Marking, src_fm: Marking, prefix: str):
        for p in src.places:
            new_p = PetriNet.Place(f"{prefix}_{p.name}")
            net.places.add(new_p)
            place_map[p] = new_p
        for t in src.transitions:
            new_t = PetriNet.Transition(f"{prefix}_{t.name}", t.label)
            net.transitions.add(new_t)
            trans_map[t] = new_t
        for arc in src.arcs:
            src_node = place_map.get(arc.source) or trans_map.get(arc.source)
            dst_node = place_map.get(arc.target) or trans_map.get(arc.target)
            petri_utils.add_arc_from_to(src_node, dst_node, net)
        for p, c in src_im.items():
            new_im[place_map[p]] = c
        for p, c in src_fm.items():
            new_fm[place_map[p]] = c

    _copy(pn_a, im_a, fm_a, "A")
    _copy(pn_b, im_b, fm_b, "B")

    return net, new_im, new_fm
