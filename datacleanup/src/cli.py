"""Enhanced CLI with hybrid AI cost savings and infrastructure guidance."""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.config.costs import get_cost_config, calculate_costs
from src.services.categorizer import HybridCategorizer
from src.services.cleaner import DataCleaner
from src.services.pdf_generator import PDFReportGenerator
from src.io.readers import FileReader
from src.io.writers import FileWriter
from src.io.schemas import CleanConfig
from src.core.logging import setup_logging
from src.core.metrics import increment_counter, set_gauge

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser with cost savings options."""
    parser = argparse.ArgumentParser(
        description="Data Cleanup Micro-SaaS with Hybrid AI Cost Savings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""