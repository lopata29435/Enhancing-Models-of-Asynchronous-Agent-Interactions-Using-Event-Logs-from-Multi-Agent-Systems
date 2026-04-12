import pm4py
import pandas as pd
from pm4py.objects.log.obj import EventLog

class XESLogParser:
    @staticmethod
    def load_log(log_path: str) -> EventLog:
        print(f"Loading log from {log_path}...")
        try:
            log = pm4py.read_xes(log_path, return_legacy_log_object=True)
        except TypeError:
            log_data = pm4py.read_xes(log_path)
            if isinstance(log_data, pd.DataFrame):
                print("Converted pandas DataFrame to EventLog for processing...")
                from pm4py.objects.conversion.log import converter as log_converter
                log = log_converter.apply(log_data, variant=log_converter.Variants.TO_EVENT_LOG)
            else:
                log = log_data
        return log

    @staticmethod
    def identify_agents(event) -> list:
        raw = event.get("org:resource", None)
        if raw is None:
            return []
        try:
            import math
            if isinstance(raw, float) and math.isnan(raw):
                return []
        except Exception:
            pass
        val = str(raw).strip()
        if not val or val.lower() in {"unknown", "nan", "none"}:
            return []
        return [a.strip() for a in val.split("+") if a.strip() and a.strip().lower() != "nan"]
