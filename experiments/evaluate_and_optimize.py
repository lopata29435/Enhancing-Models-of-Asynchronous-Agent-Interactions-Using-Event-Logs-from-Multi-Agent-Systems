import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import argparse
import copy
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.petri_net.obj import PetriNet, Marking

from log_parser import XESLogParser
from synthesis_algorithm import MultiAgentPetriNetSynthesizer

def evaluate_metrics(log, net, im, fm):
    try:
        replay_result = token_replay.apply(log, net, im, fm)
        trace_fitnesses = [res['trace_fitness'] for res in replay_result]
        avg_fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0
    except Exception as e:
        print(f"Warning: Could not compute fitness: {e}")
        avg_fitness = 0.0

    try:
        precision = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ETCONFORMANCE_TOKEN)
    except Exception as e:
        print(f"Warning: Could not compute precision: {e}")
        precision = 0.0
        
    return avg_fitness, precision

def remove_place(net, im, fm, place_name):
    target_place = None
    for p in net.places:
        if p.name == place_name:
            target_place = p
            break
    if not target_place: 
        return None, [], [], 0, 0
    
    in_arcs = list(target_place.in_arcs)
    out_arcs = list(target_place.out_arcs)
    
    for arc in in_arcs:
        if arc in arc.source.out_arcs:
            arc.source.out_arcs.remove(arc)
        if arc in net.arcs:
            net.arcs.remove(arc)
            
    for arc in out_arcs:
        if arc in arc.target.in_arcs:
            arc.target.in_arcs.remove(arc)
        if arc in net.arcs:
            net.arcs.remove(arc)
            
    net.places.remove(target_place)
    
    init_tokens = im.get(target_place, 0)
    final_tokens = fm.get(target_place, 0)
    
    if target_place in im: del im[target_place]
    if target_place in fm: del fm[target_place]
    
    return target_place, in_arcs, out_arcs, init_tokens, final_tokens

def restore_place(net, im, fm, place, in_arcs, out_arcs, init_tokens, final_tokens):
    net.places.add(place)
    for arc in in_arcs:
        arc.source.out_arcs.add(arc)
        net.arcs.add(arc)
    for arc in out_arcs:
        arc.target.in_arcs.add(arc)
        net.arcs.add(arc)
        
    if init_tokens > 0:
        im[place] = init_tokens
    if final_tokens > 0:
        fm[place] = final_tokens

def main():
    parser = argparse.ArgumentParser(description="Evaluate and optimize Multi-Agent Petri Net")
    parser.add_argument("log", help="Path to XES event log")
    args = parser.parse_args()

    log_path = args.log
    if not os.path.exists(log_path):
        print(f"Error: File {log_path} not found.")
        sys.exit(1)

    print(f"Loading log from {log_path}...")
    log = pm4py.read_xes(log_path)
    
    print("\n[1] Synthesizing Initial MAS Model...")
    synthesizer = MultiAgentPetriNetSynthesizer(log)
    global_net, global_im, global_fm = synthesizer.run()

    print("\n[2] Evaluating Initial Global System...")
    init_sys_fitness, init_sys_precision = evaluate_metrics(log, global_net, global_im, global_fm)
    print(f" -> Initial Global Fitness:   {init_sys_fitness:.4f}")
    print(f" -> Initial Global Precision: {init_sys_precision:.4f}")

    print("\n[3] Evaluating Individual Agents...")
    agent_metrics = {}
    for agent, agent_log in synthesizer.sub_logs.items():
        local_net, local_im, local_fm = synthesizer.local_nets[agent]
        f, p = evaluate_metrics(log, local_net, local_im, local_fm)
        agent_metrics[agent] = {'fitness': f, 'precision': p}
        print(f" -> Agent {agent}: Fitness = {f:.4f}, Precision = {p:.4f}")

    if not agent_metrics:
        print("No agents found in the log.")
        sys.exit(1)

    min_agent_precision = min(v['precision'] for v in agent_metrics.values())
    print(f"\n -> Minimum Agent Precision: {min_agent_precision:.4f}")
    
    passed_hypothesis = init_sys_precision >= min_agent_precision and init_sys_fitness >= 0.9999
    print(f"\nHypothesis Check: Global Prec >= Min Agent Prec? {'YES' if init_sys_precision >= min_agent_precision else 'NO'}")
    print(f"Hypothesis Check: Global Fitness == 1.0? {'YES' if init_sys_fitness >= 0.9999 else 'NO'}")

    print("\n[4] Optimizing MAS Model (Removing Redundant Channels)...")
    channels = [p.name for p in global_net.places if p.name.startswith("chan_")]
    initial_channel_count = len(channels)
    print(f"Identified {initial_channel_count} channels.")

    current_precision = init_sys_precision
    removed_channels = 0

    for idx, chan_name in enumerate(channels, 1):
        print(f"  Testing channel {idx}/{len(channels)}: {chan_name} ... ", end="")
        p, in_a, out_a, i_tok, f_tok = remove_place(global_net, global_im, global_fm, chan_name)
        if not p:
            print("Not found.")
            continue
        
        _, new_precision = evaluate_metrics(log, global_net, global_im, global_fm)

        if new_precision >= current_precision - 1e-6:
            print(f"Removed! (New Prec: {new_precision:.4f})")
            current_precision = max(current_precision, new_precision)
            removed_channels += 1
        else:
            print(f"Kept! (Prec drop to {new_precision:.4f})")
            restore_place(global_net, global_im, global_fm, p, in_a, out_a, i_tok, f_tok)

    final_channel_count = len([p.name for p in global_net.places if p.name.startswith("chan_")])
    _, final_sys_fitness = evaluate_metrics(log, global_net, global_im, global_fm)

    print("\n" + "="*50)
    print("                SUMMARY TABLE")
    print("="*50)
    print(f"System Fitness                     : {init_sys_fitness:.4f}")
    print(f"Minimum Agent Precision            : {min_agent_precision:.4f}")
    print(f"Precision (Initial MAS Model)      : {init_sys_precision:.4f}")
    print(f"Precision (Optimized MAS Model)    : {current_precision:.4f}")
    print(f"Number of Channels (Initial)       : {initial_channel_count}")
    print(f"Number of Channels (Optimized)     : {final_channel_count}")
    print(f"Channels Removed                   : {removed_channels}")
    print("="*50)

if __name__ == "__main__":
    main()
