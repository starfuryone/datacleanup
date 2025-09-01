#!/usr/bin/env python3
# Simulated worker entrypoint
from src.worker.run_job import run_job_from_paths

if __name__ == "__main__":
    result = run_job_from_paths([])
    print("Worker finished:", result)
