import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pm4py
from pm4py.convert import convert_to_event_log
from pm4py.objects.log.obj import EventLog

from src.async_agents import compute_channel_structure
from utils.hybrid_metrics import extract_agent_interface_labels, project_hybrid_log

DATA_DIR = Path(__file__).parent.parent / "data"
SUPPORTED_PATTERNS = range(1, 8)

def _load_log(xes_path: Path) -> EventLog:
    log_or_df = pm4py.read_xes(str(xes_path))
    if not isinstance(log_or_df, EventLog):
        return convert_to_event_log(log_or_df)
    return log_or_df

def _safe_name(agent: str) -> str:
    return re.sub(r'[\\/:*?"<>|]', "-", agent).replace(" ", "_")

def _find_xes(dataset_dir: Path) -> Path | None:
    for f in sorted(dataset_dir.iterdir()):
        if f.suffix.lower() == ".xes":
            return f
    return None

def process_dataset(dataset_dir: Path, agent_attr: str = "org:resource") -> None:
    dataset = dataset_dir.name
    xes_path = _find_xes(dataset_dir)
    if xes_path is None:
        print(f"  [skip] No .xes file found in {dataset_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Dataset : {dataset}  ({xes_path.name})")

    log = _load_log(xes_path)

    (agent_a, agent_b), channels, _net_a, _net_b = compute_channel_structure(
        log, agent_attr
    )
    print(f"Agents  : {agent_a!r}  /  {agent_b!r}")
    print(f"Channels: {channels}")

    recv_a, send_a = extract_agent_interface_labels(channels, side=1)
    recv_b, send_b = extract_agent_interface_labels(channels, side=2)

    log_h1 = project_hybrid_log(log, agent_attr, agent_a, agent_b, recv_b | send_b)
    log_h2 = project_hybrid_log(log, agent_attr, agent_b, agent_a, recv_a | send_a)

    out_h1 = dataset_dir / f"{dataset}_hybrid_log_{_safe_name(agent_a)}.xes"
    out_h2 = dataset_dir / f"{dataset}_hybrid_log_{_safe_name(agent_b)}.xes"

    pm4py.write_xes(log_h1, str(out_h1))
    pm4py.write_xes(log_h2, str(out_h2))

    print(f"Saved   : {out_h1.name}  ({len(log_h1)} traces)")
    print(f"Saved   : {out_h2.name}  ({len(log_h2)} traces)")

def main() -> None:
    if not DATA_DIR.exists():
        print(f"Data directory not found: {DATA_DIR}")
        sys.exit(1)

    datasets = []
    for d in sorted(DATA_DIR.iterdir(), key=lambda p: p.name):
        if not d.is_dir():
            continue
        m = re.search(r"IP-?(\d+)$", d.name, re.IGNORECASE)
        if m and int(m.group(1)) in SUPPORTED_PATTERNS:
            datasets.append(d)

    if not datasets:
        print(f"No supported datasets (IP-1 … IP-7) found under {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(datasets)} supported dataset(s): {[d.name for d in datasets]}")

    errors = []
    for dataset_dir in datasets:
        try:
            process_dataset(dataset_dir)
        except Exception as exc:
            print(f"  [error] {dataset_dir.name}: {exc}")
            errors.append((dataset_dir.name, exc))

    print(f"\n{'='*60}")
    print(f"Done. {len(datasets) - len(errors)}/{len(datasets)} datasets processed successfully.")
    if errors:
        for name, exc in errors:
            print(f"  FAILED {name}: {exc}")

if __name__ == "__main__":
    main()
