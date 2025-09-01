"""Enhanced CLI with hybrid AI cost savings and validation/reporting support."""
from __future__ import annotations
import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional, List

import pandas as pd

from src.config.costs import get_cost_config, calculate_costs
from src.services.categorizer import HybridCategorizer
from src.services.pdf_generator import PDFReportGenerator
from src.io.readers import FileReader
from src.io.writers import FileWriter
from src.io.schemas import CleanConfig, Summary
from src.core.logging import setup_logging
from src.core.metrics import set_gauge
from src.validation import validate_dataframe

# NOTE: DataCleaner is assumed to exist in your repo
from src.services.cleaner import DataCleaner  # type: ignore

logger = logging.getLogger(__name__)

def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Data Cleanup Micro-SaaS with Hybrid AI Cost Savings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic cleaning
  python -m src.cli input.csv --out cleaned.csv

  # With hybrid AI categorization and cost analysis
  python -m src.cli input.csv --out cleaned.csv --ai-model hybrid \
    --ai-pricing-model gpt-4o --ai-input-tokens 60 --ai-output-tokens 12

  # Full deliverables bundle with PDF report
  python -m src.cli input.csv --out cleaned.csv --ai-model hybrid \
    --pdf-report report.pdf --bundle out/ --bundle-zip deliverables.zip
        """
    )

    # Input/Output
    parser.add_argument("input_file", help="Input CSV/XLSX file to clean")
    parser.add_argument("--out", "-o", help="Output cleaned CSV file path")

    # Data Processing Options
    parser.add_argument("--validate", action="store_true", help="Enable data validation and error flagging")
    parser.add_argument("--dedupe-keys", nargs="+", default=["email", "company", "phone"], help="Columns to use for deduplication")

    # AI Categorization Options
    parser.add_argument("--ai-model", choices=["hybrid", "hf", "openai"], default="hybrid", help="AI model approach")
    parser.add_argument("--categories", nargs="+",
                        default=["lead", "customer", "vendor", "expense", "other"],
                        help="Categories for AI classification")

    # Cost Analysis Parameters
    parser.add_argument("--ai-pricing-model", choices=["gpt-4o-mini", "gpt-4o"],
                        help="OpenAI model tier for cost calculations")
    parser.add_argument("--ai-input-tokens", type=int, help="Estimated input tokens per row for cost calculation")
    parser.add_argument("--ai-output-tokens", type=int, help="Estimated output tokens per row for cost calculation")

    # Reporting Options
    parser.add_argument("--report", help="JSON summary report output path")
    parser.add_argument("--pdf-report", help="PDF report output path")
    parser.add_argument("--bundle", help="Directory to create deliverables bundle")
    parser.add_argument("--bundle-zip", help="ZIP file path for deliverables bundle")

    # Configuration
    parser.add_argument("--openai-api-key", help="OpenAI API key (or use OPENAI_API_KEY env)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    return parser

def build_summary_base(rows_processed: int, duplicates_removed: int) -> dict:
    return {
        "rows_processed": rows_processed,
        "duplicates_removed": duplicates_removed,
        "data_quality_score": max(0.0, 100.0 - (duplicates_removed / max(rows_processed, 1)) * 100.0),
    }

def main():
    parser = create_parser()
    args = parser.parse_args()

    setup_logging("DEBUG" if args.verbose else "INFO")
    logger.info("Starting data cleanup")

    input_path = Path(args.input_file)
    if not input_path.exists():
        parser.error(f"Input file not found: {input_path}")

    df = FileReader.read_any(input_path)

    # Clean + dedupe
    cleaner = DataCleaner(CleanConfig())
    df_clean, duplicates_removed = cleaner.clean(df, dedupe_keys=args.dedupe_keys)

    # Optional validation
    validation_result = {"enabled": False, "stats": {}, "samples": {}}
    if args.validate:
        df_clean, counts, samples = validate_dataframe(df_clean)
        validation_result = {"enabled": True, "stats": counts, "samples": samples}

    # Categorization (hybrid costs)
    openai_key = args.openai_api_key or os.getenv("OPENAI_API_KEY")
    categorizer = HybridCategorizer(openai_api_key=openai_key)
    df_cat, cat_summary = categorizer.categorize_dataframe(
        df_clean, categories=args.categories, text_columns=None, ai_model=args.ai_model
    )
    df_clean = df_cat

    # Cost analysis
    cfg = get_cost_config()
    model = args.ai_pricing_model or cfg["ai_pricing_model"]
    itoks = args.ai_input_tokens or cfg["ai_input_tokens_per_row"]
    otoks = args.ai_output_tokens or cfg["ai_output_tokens_per_row"]
    rows_total = len(df_clean)
    rows_openai = cat_summary.get("sources", {}).get("openai", 0)
    costs = calculate_costs(rows_total=rows_total, rows_openai=rows_openai,
                            ai_pricing_model=model,
                            input_tokens_per_row=itoks,
                            output_tokens_per_row=otoks)
    cat_summary["costs"] = costs

    # Summary
    summary = build_summary_base(rows_processed=rows_total, duplicates_removed=duplicates_removed)
    summary["categorization"] = cat_summary
    summary["validation"] = validation_result
    summary["savings_summary"] = (
        f"Hybrid AI saved ${costs['savings_absolute']:.4f} "
        f"({int(costs['savings_pct']*100)}%) versus all-OpenAI on {rows_total} rows."
    )

    # Outputs
    out_path = Path(args.out or input_path.with_name(input_path.stem + "_clean.csv"))
    FileWriter.write_csv(df_clean, out_path)

    if args.report:
        FileWriter.write_json(summary, args.report)

    if args.pdf_report:
        pdf = PDFReportGenerator()
        pdf.generate_report(summary, args.pdf_report, original_filename=input_path.name)

    if args.bundle or args.bundle_zip:
        bundle_dir = Path(args.bundle or out_path.parent / "bundle")
        bundle_dir.mkdir(parents=True, exist_ok=True)

        # Always include original + cleaned + JSON summary
        summary_json_path = bundle_dir / "summary.json"
        FileWriter.write_json(summary, summary_json_path)

        files = {
            f"original/{input_path.name}": str(input_path),
            f"cleaned/{out_path.name}": str(out_path),
            "summary/summary.json": str(summary_json_path),
        }
        if args.pdf_report:
            files["report/report.pdf"] = args.pdf_report

        zip_path = Path(args.bundle_zip or (bundle_dir / "deliverables.zip"))
        FileWriter.bundle_zip(files, zip_path)

    set_gauge("data_rows", float(rows_total))
    logger.info("Done")

if __name__ == "__main__":
    main()
