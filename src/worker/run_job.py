# Placeholder worker that would load source files, run cleaning pipeline, and store outputs
# Replace with real queue/cron wiring later.
from src.services.cleaning import clean_dataframe
from src.services.formatting import normalize_dataframe
from src.services.dedupe import dedupe_dataframe

def run_job_from_paths(paths):
    # TODO: load files into DataFrames, merge, process, export
    # This is a stub for local testing
    return {"rows_processed": 0, "result_key": None}
