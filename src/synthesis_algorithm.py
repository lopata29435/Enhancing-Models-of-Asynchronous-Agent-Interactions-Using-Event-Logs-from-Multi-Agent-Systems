import math
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple

def localize_acyclic_interaction(traces: List[List[str]], agent1_events: List[str], agent2_events: List[str]) -> List[Tuple[str, str]]:
    if not agent1_events or not agent2_events:
        return []

    matrix = np.full((len(agent1_events), len(agent2_events)), fill_value=2)

    for trace in traces:
        for i in range(len(trace) - 1):
            for j in range(i + 1, len(trace)):
                if trace[i] in agent1_events and trace[j] in agent1_events:
                    continue
                if trace[i] in agent2_events and trace[j] in agent2_events:
                    continue

                if trace[i] in agent1_events and trace[j] in agent2_events:
                    row = agent1_events.index(trace[i])
                    col = agent2_events.index(trace[j])
                    if matrix[row, col] == 2:
                        matrix[row, col] = -1
                    if matrix[row, col] == -1 or matrix[row, col] == 0:
                        continue
                    if matrix[row, col] == 1:
                        matrix[row, col] = 0
                elif trace[i] in agent2_events and trace[j] in agent1_events:
                    row = agent1_events.index(trace[j])
                    col = agent2_events.index(trace[i])
                    if matrix[row, col] == 2:
                        matrix[row, col] = 1
                    if matrix[row, col] == 1 or matrix[row, col] == 0:
                        continue
                    if matrix[row, col] == -1:
                        matrix[row, col] = 0

    minimum = []
    
    for i in range(len(agent1_events)):
        for j in range(len(agent2_events)):
            try:
                if matrix[i, j] == -1 and matrix[i+1, j] != -1 and matrix[i, j-1] != -1:
                    minimum.append((agent1_events[i], agent2_events[j]))
                if matrix[i, j] == 1 and matrix[i-1, j] != 1 and matrix[i, j+1] != 1:
                    minimum.append((agent2_events[j], agent1_events[i]))
            except IndexError:
                continue

    return minimum

def localize_cyclic_interaction(traces: List[List[str]], unique_a: List[str], unique_b: List[str]) -> List[Dict]:
    if not unique_a or not unique_b:
        return []

    mins = np.full((len(unique_a), len(unique_b)), fill_value=1000)
    maxs = np.full((len(unique_a), len(unique_b)), fill_value=-1000)
    
    for trace in traces:
        for i in range(len(unique_a)):
            for j in range(len(unique_b)):
                max_k = 0
                min_k = 0
                current = 0
                for t in trace:
                    if t == unique_a[i]:
                        current += 1
                    if t == unique_b[j]:
                        current -= 1
                    max_k = np.max([current, max_k])
                    min_k = np.min([current, min_k])
                mins[i][j] = min(min_k, mins[i][j])
                maxs[i][j] = max(max_k, maxs[i][j])

    results = []
    
    for i in range(len(unique_a)):
        for j in range(len(unique_b)):
            min_val = mins[i][j]
            max_val = maxs[i][j]
            
            if min_val != 1000 and max_val != -1000:
                capacity = max_val - min_val
                initial_tokens = abs(min_val) if min_val < 0 else 0
                
                if capacity > 0:
                    results.append({
                        "sender": unique_a[i],
                        "receiver": unique_b[j],
                        "global_min": min_val,
                        "global_max": max_val,
                        "capacity": capacity,
                        "initial_tokens": initial_tokens
                    })

    return results
