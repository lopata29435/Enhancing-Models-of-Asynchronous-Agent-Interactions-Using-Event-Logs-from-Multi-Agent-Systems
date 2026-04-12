import os
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../src'))
import copy
import glob
import pm4py
from pm4py.algo.conformance.tokenreplay import algorithm as token_replay
from pm4py.algo.evaluation.precision import algorithm as precision_evaluator
from pm4py.objects.petri_net.utils import petri_utils

from log_parser import XESLogParser
from synthesis_algorithm import MultiAgentPetriNetSynthesizer
from compute_fitness import generate_final_marking_from_net

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
                trace_fitnesses.append(max(0.0, 1.0 - (missing + remaining) / denom))
        else:
            trace_fitnesses.append(float(r))
            
    fitness = sum(trace_fitnesses) / len(trace_fitnesses) if trace_fitnesses else 0.0

    try:
        precision = precision_evaluator.apply(log, net, im, fm, variant=precision_evaluator.Variants.ALIGN_ETCONFORMANCE)
    except Exception as e:
        print(f"Warning: Could not compute precision: {e}")
        precision = 0.0

    return fitness, precision

def copy_petri_net(net, im, fm):
    net_c = copy.deepcopy(net)
    im_c = pm4py.objects.petri_net.obj.Marking()
    fm_c = pm4py.objects.petri_net.obj.Marking()
    
    for p, weight in im.items():
        for p_c in net_c.places:
            if p_c.name == p.name:
                im_c[p_c] = weight
    for p, weight in fm.items():
        for p_c in net_c.places:
            if p_c.name == p.name:
                fm_c[p_c] = weight
                
    return net_c, im_c, fm_c

def run_evaluation_for_dataset(dataset_name):
    data_path = os.path.join("data", dataset_name)
    log_files = glob.glob(os.path.join(data_path, "*_init_log_enriched.xes"))
    if not log_files:
        log_files = glob.glob(os.path.join(data_path, "*.xes"))
    log_files = [f for f in log_files if "__MACOSX" not in f and not os.path.basename(f).startswith('.')]
    
    if not log_files:
        print(f"No valid .xes log file found in {data_path}")
        return
        
    log_file = log_files[0]
    print(f"\n[INFO] Loading log: {log_file}")
    
    try:
        log = XESLogParser.load_log(log_file)
    except Exception as e:
        print("Failed to load log:", e)
        return

    print("[INFO] Running Synthesis Algorithm to gather all nets...")
    synthesizer = MultiAgentPetriNetSynthesizer(log)
    sys_net, sys_im, sys_fm = synthesizer.run()
    
    agent_precisions = {}
    print("\n[INFO] Evaluating Individual Agents...")
    for agent, agent_log in synthesizer.sub_logs.items():
        agent_net, agent_im, agent_fm = synthesizer.local_nets[agent]
        f, p = eval_net(agent_log, agent_net, agent_im, agent_fm)
        agent_precisions[agent] = p
        print(f"  Agent {agent}: Fitness = {f:.4f}, Precision = {p:.4f}")
        
    min_agent_prec = min(agent_precisions.values()) if agent_precisions else 0.0
    
    print("\n[INFO] Evaluating Initial System Model (Maximal)...")
    init_channels = [p for p in sys_net.places if p.name.startswith("chan_")]
    init_channel_count = len(init_channels)
    sys_f, sys_p = eval_net(log, sys_net, sys_im, sys_fm)
    print(f"  System Fitness: {sys_f:.4f}")
    print(f"  System Precision: {sys_p:.4f}")
    print(f"  Channels Count: {init_channel_count}")
    
    print("\n[INFO] Optimizing Redundant Channels...")
    current_net, current_im, current_fm = sys_net, sys_im, sys_fm
    current_p = sys_p
    current_f = sys_f
    
    channels_to_check = [p.name for p in init_channels]
    removed_channels = 0
    
    for ch_name in channels_to_check:
        print(f"   -> Testing removal of {ch_name}...")
        
        test_net, test_im, test_fm = copy_petri_net(current_net, current_im, current_fm)
        p_to_remove = next((p for p in test_net.places if p.name == ch_name), None)
        
        if not p_to_remove:
            continue
            
        petri_utils.remove_place(test_net, p_to_remove)
        
        test_f, test_p = eval_net(log, test_net, test_im, test_fm)
        
        if test_p >= current_p and test_f >= current_f:
            print(f"      [REMOVED] Precision {test_p:.4f} >= {current_p:.4f}, Fitness {test_f:.4f}")
            current_net, current_im, current_fm = test_net, test_im, test_fm
            current_p = test_p
            current_f = test_f
            removed_channels += 1
        else:
            print(f"      [KEPT] Precision {test_p:.4f} < {current_p:.4f} or Fitness dropped ({test_f:.4f})")
            
    final_channel_count = init_channel_count - removed_channels
    print("\n" + "="*60)
    print("SUMMARY REPORT")
    print("="*60)
    print(f"{'Metric':<40} | {'Value'}")
    print("-" * 60)
    print(f"{'Fitness (System)':<40} | {current_f:.4f}")
    print(f"{'Precision (Individual Agents Min)':<40} | {min_agent_prec:.4f}")
    print(f"{'Precision (Initial MAS Model)':<40} | {sys_p:.4f}")
    print(f"{'Precision (Optimized MAS Model)':<40} | {current_p:.4f}")
    print(f"{'Channels (Initial)':<40} | {init_channel_count}")
    print(f"{'Channels (Optimized)':<40} | {final_channel_count}")
    print("-" * 60)
    
    if current_p >= min_agent_prec:
        print("=> HYPOTHESIS CONFIRMED: System Precision >= Minimum Agent Precision")
    else:
        print("=> HYPOTHESIS FAILED: System Precision < Minimum Agent Precision")
        
    out_dir = os.path.join("output", dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{dataset_name}_composition_optimized.pnml")
    pm4py.write_pnml(current_net, current_im, current_fm, out_file)
    print(f"\n[INFO] Optimized model saved to {out_file}")

def select_dataset(data_dir="data"):
    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found.")
        return None
    
    datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.startswith('.')]
    datasets.sort(key=lambda x: int(x.split('-')[1]) if '-' in x and x.split('-')[1].isdigit() else 0)
    
    if not datasets:
        print("No datasets found.")
        return None
        
    print("\n" + "="*40)
    print("Available datasets for evaluation:")
    for i, ds in enumerate(datasets):
        print(f"{i + 1}. {ds}")
    print("="*40)
        
    while True:
        choice = input("\nSelect a dataset number to test (or 'q' to quit): ").strip()
        if choice.lower() == 'q':
            return None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(datasets):
                return datasets[idx]
            else:
                print("Invalid number. Try again.")
        except ValueError:
            print("Please enter a valid number.")

def main():
    if len(sys.argv) > 1 and sys.argv[1] != "--interactive":
        import argparse
        parser = argparse.ArgumentParser(description="Evaluate and Optimize MAS Petri Nets")
        parser.add_argument("dataset", help="Name of the dataset in data folder (e.g. IP-1)")
        args = parser.parse_args()
        
        run_evaluation_for_dataset(args.dataset)
        return
        
    while True:
        dataset = select_dataset("data")
        if dataset:
            run_evaluation_for_dataset(dataset)
            print("\n" + "="*50)
        else:
             print("Exiting...")
             break

if __name__ == "__main__":
    main()
