import pm4py
import os

DATA_DIR = "data"

AGENT_MAPPING = {
    "IP-2": {"t": "Agent 1", "q": "Agent 2"},
    "IP-3": {"t": "Agent 1", "q": "Agent 2"},
    "IP-5": {"t": "Agent 1", "q": "Agent 2"},
    "IP-6": {"t": "Agent 1", "q": "Agent 2"},
    "IP-7": {"q": "Agent 1", "t": "Agent 2"},
    "IP-8": {"t": "Agent 1", "r": "Agent 2", "q": "Agent 3"},
    "IP-9": {"t": "Agent 1", "q": "Agent 2"},
    "IP-10": {},
    "IP-11": {"q": "Agent 1", "t": "Agent 2"},
    "IP-12": {"t": "Agent 1", "q": "Agent 2"}
}

RULES = {
    "IP-2": {
        "Agent 1": {"a!_1", "a!_2", "b!"},
        "Agent 2": {"a?", "b?_1", "b?_2"}
    },
    "IP-3": {
        "Agent 1": {"a!", "b!"},
        "Agent 2": {"a?", "b?"}
    },
    "IP-5": {
        "Agent 1": {"a!", "b!_1", "b!_2", "c?", "d?_1", "d?_2"},
        "Agent 2": {"a?_1", "a?_2", "b?", "d!", "c!"}
    },
    "IP-6": {
        "Agent 1": {"a!", "b!_1", "b!_2", "c?", "d?_1", "d?_2"},
        "Agent 2": {"c!_1", "c!_2", "d!"}
    },
    "IP-7": {
        "Agent 1": {"a!_1", "a!_2", "b?_1", "b?_2", "c?_1", "c?_2"},
        "Agent 2": {"a?_1", "a?_2", "c!_1", "c!_2", "b!_1", "b!_2"}
    },
    "IP-8": {
        "Agent 1": {"a!_1", "a!_2", "ackA?", "bR?", "a?_u"},
        "Agent 2": {"b!_1", "b!_2", "aR?", "ackB?", "b?_u"},
        "Agent 3": {"a?", "b?", "ackA!", "aR!", "ackB!", "bR?"}
    },
    "IP-9": {
        "Agent 1+Agent 2": {"s_1", "s_2", "s_3", "s_4"},
        "Agent 1": {"a!_1", "a!_2", "b?_1", "b?_2"},
        "Agent 2": {"a?", "b!_2", "b!_1"}
    },
    "IP-10": {
        "Agent 1+Agent 2": {"s_1", "s_2"},
        "_keep_existing": True
    },
    "IP-11": {
        "Agent 1": {"a?_1", "a?_2", "b!_1", "b!_2"},
        "Agent 2": {"a!_1", "a!_2", "b?"},
        "Agent 1+Agent 2": {"s"}
    },
    "IP-12": {
        "Agent 1": {"a!", "b?"},
        "Agent 2": {"b!_1", "b!_2", "a?"},
        "Agent 1+Agent 2": {"s"}
    }
}

def enrich_log(log_path, rules, dataset_name):
    print(f"Enriching {dataset_name} ({log_path})...")
    try:
        log = pm4py.read_xes(log_path, return_legacy_log_object=True)
    except Exception as e:
        print(f"Error reading {log_path}: {e}")
        return

    event_to_agent = {}
    for agent, events in rules.items():
        if agent.startswith("_"): continue
        for event_name in events:
            event_to_agent[event_name] = agent

    keep_existing = rules.get("_keep_existing", False)
    mapping = AGENT_MAPPING.get(dataset_name, {})

    count_updated = 0

    for trace in log:
        for event in trace:
            name = event.get("concept:name", "")

            if name in event_to_agent:
                event["org:resource"] = event_to_agent[name]
                count_updated += 1
            else:
                if not keep_existing:
                    assigned = False
                    for prefix, agent_label in mapping.items():
                        if name.startswith(prefix):
                            event["org:resource"] = agent_label
                            count_updated += 1
                            assigned = True
                            break
                    
                    if not assigned and mapping:
                        pass

    if count_updated > 0:
        print(f"Updated {count_updated} events. Saving...")
        new_log_path = log_path.replace(".xes", "_enriched.xes")
        try:
            pm4py.write_xes(log, new_log_path)
            print("Done.")
        except Exception as e:
            print(f"Error writing {new_log_path}: {e}")
    else:
        print("No matches found or no changes needed.")

def main():
    print(f"Scanning {DATA_DIR}...")
    try:
        entries = os.listdir(DATA_DIR)
    except FileNotFoundError:
        print(f"Directory not found: {DATA_DIR}")
        return

    for entry in entries:
        full_path = os.path.join(DATA_DIR, entry)
        if os.path.isdir(full_path) and entry in RULES:
            dataset_key = entry
            for root, _, files in os.walk(full_path):
                for file in files:
                    if file.endswith(".xes") and not file.endswith("_enriched.xes"):
                        log_path = os.path.join(root, file)
                        enrich_log(log_path, RULES[dataset_key], dataset_key)

if __name__ == "__main__":
    main()
