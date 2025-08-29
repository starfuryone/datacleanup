"""Enhanced API routes with hybrid AI cost savings and deliverables bundling."""

import asyncio
import io
import json
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import List, Optional

import pandas as pd
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, Field

from src.config.costs import get_cost_config, calculate_costs
from src.services.categorizer import HybridCategorizer
from src.services.cleaner import DataCleaner
from src.services.pdf_generator import PDFReportGenerator
from src.io.readers import FileReader
from src.io.writers import FileWriter
from src.io.schemas import CleanConfig
from src.core.metrics import increment_counter, set_gauge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["data-cleaning"])

class CleanResponse(BaseModel):
    """Response model for clean operation summary."""
    processing: dict
    cleaning: dict
    categorization: dict
    savings_summary: Optional[str] = None

async def validate_file(file: UploadFile) -> None:
    """Validate uploaded file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    # Check file extension
    allowed_extensions = {'.csv', '.xlsx', '.xls'}
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400, 
            detail=f"Unsupported file type: {file_ext}. Allowed: {', '.join(allowed_extensions)}"
        )
    
    # Check file size (limit to 50MB)
    MAX_SIZE = 50 * 1024 * 1024  # 50MB
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    if file_size > MAX_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {file_size/1024/1024:.1f}MB. Maximum: {MAX_SIZE/1024/1024}MB"
        )

@router.post("/clean", response_class=StreamingResponse)
async def clean_data(
    file: UploadFile = File(...),
    ai_model: str = Form("hybrid"),
    categories: Optional[str] = Form(None),  # JSON string of list
    validate: bool = Form(False),
    dedupe_keys: Optional[str] = Form(None),  # JSON string of list
    ai_pricing_model: Optional[str] = Form(None),
    ai_input_tokens: Optional[int] = Form(None),
    ai_output_tokens: Optional[int] = Form(None),
    want_pdf: bool = Form(False),
    bundle: Optional[str] = Form(None),  # "zip" or "json" for bundle format
) -> StreamingResponse:
    """
    Clean data with hybrid AI categorization and optional cost analysis.
    
    Args:
        file: CSV/XLSX file to clean
        ai_model: "hybrid", "hf", "openai", or "none"
        categories: JSON string of categories list
        validate: Enable data validation
        dedupe_keys: JSON string of deduplication keys
        ai_pricing_model: "gpt-4o-mini" or "gpt-4o"
        ai_input_tokens: Input tokens per row assumption
        ai_output_tokens: Output tokens per row assumption
        want_pdf: Include PDF report in response
        bundle: "zip" for ZIP bundle, "json" for metadata only
    
    Returns:
        Cleaned CSV file or ZIP bundle with deliverables
    """
    await validate_file(file)
    
    try:
        # Parse JSON parameters
        categories_list = json.loads(categories) if categories else ["lead", "customer", "vendor", "expense", "other"]
        dedupe_keys_list = json.loads(dedupe_keys) if dedupe_keys else ["email", "company", "phone"]
        
        # Get cost configuration
        cost_config = get_cost_config()
        ai_pricing_model = ai_pricing_model or cost_config["ai_pricing_model"]
        ai_input_tokens = ai_input_tokens or cost_config["ai_input_tokens_per_row"]
        ai_output_tokens = ai_output_tokens or cost_config["ai_output_tokens_per_row"]
        
        # Read uploaded file
        file_content = await file.read()
        reader = FileReader()
        
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp.flush()
            df = await reader.read_file(tmp.name)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found in uploaded file")
        
        original_rows = len(df)
        logger.info(f"API processing {original_rows:,} rows from {file.filename}")
        
        # Clean data
        config = CleanConfig(
            dedupe_keys=dedupe_keys_list,
            validate=validate,
        )
        cleaner = DataCleaner(config)
        df_cleaned, clean_summary = await cleaner.clean_dataframe(df)
        
        # AI Categorization with cost tracking
        categorization_summary = {"enabled": False}
        
        if ai_model != "none":
            categorizer = HybridCategorizer(
                confidence_threshold=0.7,
            )
            
            df_categorized, categorization_summary = await categorizer.categorize_dataframe(
                df_cleaned, categories_list, ai_model=ai_model
            )
            df_cleaned = df_categorized
            
            # Calculate cost savings
            rows_total = categorization_summary.get("rows_processed", 0)
            rows_openai = categorization_summary.get("sources", {}).get("openai", 0)
            
            costs_data = calculate_costs(
                rows_total=rows_total,
                rows_openai=rows_openai,
                ai_pricing_model=ai_pricing_model,
                input_tokens_per_row=ai_input_tokens,
                output_tokens_per_row=ai_output_tokens,
            )
            
            categorization_summary["costs"] = costs_data
            
            # Update metrics
            set_gauge("ai_rows_total", rows_total, {"ai_pricing_model": ai_pricing_model})
            set_gauge("ai_rows_hf_total", costs_data["rows_breakdown"]["hf"])
            set_gauge("ai_rows_openai_total", costs_data["rows_breakdown"]["openai"])
            set_gauge("ai_cost_all_openai_usd", costs_data["cost_all_openai"])
            set_gauge("ai_cost_hybrid_usd", costs_data["cost_hybrid"])
            set_gauge("ai_savings_usd", costs_data["savings_absolute"])
        
        # Create response summary
        summary = {
            "processing": {
                "input_file": file.filename,
                "timestamp": pd.Timestamp.now().isoformat(),
                "rows_original": original_rows,
                "rows_processed": len(df_cleaned),
                "rows_removed": original_rows - len(df_cleaned),
            },
            "cleaning": clean_summary,
            "categorization": categorization_summary,
        }
        
        # Add cost savings summary
        if categorization_summary.get("enabled") and "costs" in categorization_summary:
            costs = categorization_summary["costs"]
            summary["savings_summary"] = (
                f"Using {ai_model} categorization saved ${costs['savings_absolute']:.4f} "
                f"(~{costs['savings_pct']*100:.0f}%) compared to sending all rows to OpenAI "
                f"(model={costs['model']})"
            )
        
        # Return appropriate response format
        if bundle == "zip":
            # Create ZIP bundle with all deliverables
            zip_buffer = io.BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zipf:
                # Original file
                zipf.writestr(f"01_original_{file.filename}", file_content)
                
                # Cleaned CSV
                cleaned_csv = io.StringIO()
                df_cleaned.to_csv(cleaned_csv, index=False)
                zipf.writestr(f"02_cleaned_{Path(file.filename).stem}.csv", cleaned_csv.getvalue())
                
                # Summary JSON
                zipf.writestr("03_summary.json", json.dumps(summary, indent=2, default=str))
                
                # PDF report if requested
                if want_pdf:
                    pdf_generator = PDFReportGenerator()
                    with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as pdf_tmp:
                        pdf_generator.generate_report(
                            summary_data=summary,
                            output_path=pdf_tmp.name,
                            original_filename=file.filename
                        )
                        
                        with open(pdf_tmp.name, 'rb') as pdf_file:
                            zipf.writestr("04_executive_report.pdf", pdf_file.read())
            
            zip_buffer.seek(0)
            
            increment_counter("api_requests_total", {"endpoint": "clean", "format": "zip"})
            
            return StreamingResponse(
                io.BytesIO(zip_buffer.read()),
                media_type="application/zip",
                headers={"Content-Disposition": f"attachment; filename=cleaned_{Path(file.filename).stem}_deliverables.zip"}
            )
            
        else:
            # Return just the cleaned CSV
            csv_buffer = io.StringIO()
            df_cleaned.to_csv(csv_buffer, index=False)
            
            increment_counter("api_requests_total", {"endpoint": "clean", "format": "csv"})
            
            # Include summary in response headers for programmatic access
            headers = {
                "Content-Disposition": f"attachment; filename=cleaned_{Path(file.filename).stem}.csv",
                "X-Processing-Summary": json.dumps({
                    "rows_original": original_rows,
                    "rows_cleaned": len(df_cleaned),
                    "ai_model": ai_model,
                    "savings_summary": summary.get("savings_summary", ""),
                })
            }
            
            return StreamingResponse(
                io.BytesIO(csv_buffer.getvalue().encode()),
                media_type="text/csv",
                headers=headers
            )
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {e}")
    except Exception as e:
        logger.error(f"API cleaning failed: {e}")
        increment_counter("api_errors_total", {"endpoint": "clean"})
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@router.post("/clean:report", response_model=CleanResponse)
async def get_clean_report(
    file: UploadFile = File(...),
    ai_model: str = Form("hybrid"),
    categories: Optional[str] = Form(None),
    validate: bool = Form(False),
    dedupe_keys: Optional[str] = Form(None),
    ai_pricing_model: Optional[str] = Form(None),
    ai_input_tokens: Optional[int] = Form(None),
    ai_output_tokens: Optional[int] = Form(None),
) -> CleanResponse:
    """
    Get processing summary report without returning cleaned data.
    
    Useful for cost estimation and analysis before committing to full processing.
    """
    await validate_file(file)
    
    try:
        # Parse parameters (same as clean endpoint)
        categories_list = json.loads(categories) if categories else ["lead", "customer", "vendor", "expense", "other"]
        dedupe_keys_list = json.loads(dedupe_keys) if dedupe_keys else ["email", "company", "phone"]
        
        cost_config = get_cost_config()
        ai_pricing_model = ai_pricing_model or cost_config["ai_pricing_model"]
        ai_input_tokens = ai_input_tokens or cost_config["ai_input_tokens_per_row"]
        ai_output_tokens = ai_output_tokens or cost_config["ai_output_tokens_per_row"]
        
        # Process file (same logic as clean endpoint)
        file_content = await file.read()
        reader = FileReader()
        
        with tempfile.NamedTemporaryFile(suffix=Path(file.filename).suffix, delete=False) as tmp:
            tmp.write(file_content)
            tmp.flush()
            df = await reader.read_file(tmp.name)
        
        if df.empty:
            raise HTTPException(status_code=400, detail="No data found in uploaded file")
        
        original_rows = len(df)
        
        # Quick cleaning for analysis
        config = CleanConfig(dedupe_keys=dedupe_keys_list, validate=validate)
        cleaner = DataCleaner(config)
        df_cleaned, clean_summary = await cleaner.clean_dataframe(df)
        
        # Cost estimation (without actually running AI categorization)
        categorization_summary = {"enabled": False}
        
        if ai_model != "none":
            # Estimate costs based on row count
            rows_total = len(df_cleaned)
            
            # For estimation, assume different fallback rates based on model
            if ai_model == "hybrid":
                estimated_openai_rate = 0.20  # 20% fallback to OpenAI
            elif ai_model == "openai":
                estimated_openai_rate = 1.0   # 100% OpenAI
            else:  # hf
                estimated_openai_rate = 0.0   # 0% OpenAI
            
            rows_openai_estimated = int(rows_total * estimated_openai_rate)
            
            costs_data = calculate_costs(
                rows_total=rows_total,
                rows_openai=rows_openai_estimated,
                ai_pricing_model=ai_pricing_model,
                input_tokens_per_row=ai_input_tokens,
                output_tokens_per_row=ai_output_tokens,
            )
            
            categorization_summary = {
                "enabled": True,
                "rows_processed": rows_total,
                "estimated": True,
                "costs": costs_data,
                "model_config": {
                    "ai_model": ai_model,
                    "estimated_openai_rate": estimated_openai_rate,
                }
            }
        
        # Create summary
        summary = {
            "processing": {
                "input_file": file.filename,
                "timestamp": pd.Timestamp.now().isoformat(),
                "rows_original": original_rows,
                "rows_processed": len(df_cleaned),
                "rows_removed": original_rows - len(df_cleaned),
                "analysis_mode": True,
            },
            "cleaning": clean_summary,
            "categorization": categorization_summary,
        }
        
        # Add estimated savings summary
        if categorization_summary.get("enabled") and "costs" in categorization_summary:
            costs = categorization_summary["costs"]
            summary["savings_summary"] = (
                f"Estimated: {ai_model} categorization could save ${costs['savings_absolute']:.4f} "
                f"(~{costs['savings_pct']*100:.0f}%) vs all-OpenAI (model={costs['model']})"
            )
        
        increment_counter("api_requests_total", {"endpoint": "clean:report"})
        
        return CleanResponse(**summary)
    
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON in parameters: {e}")
    except Exception as e:
        logger.error(f"API report generation failed: {e}")
        increment_counter("api_errors_total", {"endpoint": "clean:report"})
        raise HTTPException(status_code=500, detail=f"Report generation failed: {str(e)}")

@router.get("/cost-estimate")
async def get_cost_estimate(
    rows: int,
    ai_model: str = "hybrid",
    ai_pricing_model: str = "gpt-4o-mini",
    ai_input_tokens: int = 50,
    ai_output_tokens: int = 10,
) -> dict:
    """
    Get cost estimate for processing without uploading a file.
    
    Useful for budgeting and planning before processing large datasets.
    """
    if rows <= 0 or rows > 100000:  # Reasonable limits
        raise HTTPException(
            status_code=400, 
            detail="Row count must be between 1 and 100,000"
        )
    
    try:
        # Estimate OpenAI usage based on model type
        if ai_model == "hybrid":
            estimated_openai_rate = 0.20  # 20% fallback
        elif ai_model == "openai":
            estimated_openai_rate = 1.0   # 100% OpenAI
        elif ai_model == "hf":
            estimated_openai_rate = 0.0   # 0% OpenAI
        else:
            estimated_openai_rate = 0.20  # Default to hybrid
        
        rows_openai_estimated = int(rows * estimated_openai_rate)
        
        costs_data = calculate_costs(
            rows_total=rows,
            rows_openai=rows_openai_estimated,
            ai_pricing_model=ai_pricing_model,
            input_tokens_per_row=ai_input_tokens,
            output_tokens_per_row=ai_output_tokens,
        )
        
        response = {
            "estimate": costs_data,
            "parameters": {
                "rows": rows,
                "ai_model": ai_model,
                "estimated_openai_rate": estimated_openai_rate,
            },
            "savings_summary": (
                f"Estimated: {ai_model} approach could save ${costs_data['savings_absolute']:.4f} "
                f"(~{costs_data['savings_pct']*100:.0f}%) vs all-OpenAI"
            ),
            "disclaimer": "Estimates based on assumed fallback rates. Actual costs may vary based on data complexity."
        }
        
        increment_counter("api_requests_total", {"endpoint": "cost-estimate"})
        
        return response
    
    except Exception as e:
        logger.error(f"Cost estimation failed: {e}")
        increment_counter("api_errors_total", {"endpoint": "cost-estimate"})
        raise HTTPException(status_code=500, detail=f"Cost estimation failed: {str(e)}")

@router.get("/infrastructure-requirements")
async def get_infrastructure_requirements(
    concurrent_users: int = 1000,
    include_costs: bool = False,
) -> dict:
    """
    Get infrastructure recommendations for specified concurrent users.
    
    Provides server requirements, scaling guidance, and optional cost estimates.
    """
    if concurrent_users <= 0 or concurrent_users > 50000:
        raise HTTPException(
            status_code=400,
            detail="Concurrent users must be between 1 and 50,000"
        )
    
    try:
        # Calculate infrastructure requirements
        base_cpu_cores = max(8, concurrent_users // 125)
        base_ram_gb = max(16, concurrent_users // 62.5)
        base_storage_gb = max(500, concurrent_users * 0.5)
        
        recommendations = {
            "concurrent_users": concurrent_users,
            "server_requirements": {
                "os": "Ubuntu 22.04 LTS or CentOS Stream 9",
                "cpu_cores": f"{base_cpu_cores} vCPU cores minimum",
                "ram": f"{int(base_ram_gb)} GB RAM minimum",
                "storage": f"{int(base_storage_gb)} GB SSD storage",
                "network": "1 Gbps network connection minimum"
            },
            "scaling": {
                "load_balancer": "Required" if concurrent_users >= 500 else "Optional",
                "orchestration": "Kubernetes" if concurrent_users >= 500 else "Docker Compose",
                "replicas": f"{max(3, concurrent_users // 300)} app replicas minimum" if concurrent_users >= 500 else "2-3 replicas",
                "database": "PostgreSQL with read replicas" if concurrent_users >= 500 else "PostgreSQL single instance",
                "caching": "Redis cluster" if concurrent_users >= 500 else "Single Redis instance"
            },
            "additional_services": {}
        }
        
        # Add additional services for high-scale deployments
        if concurrent_users >= 2000:
            recommendations["additional_services"].update({
                "cdn": "Required (CloudFlare, AWS CloudFront)",
                "monitoring": "Full observability stack (Prometheus, Grafana, ELK)",
                "backup": "Real-time backup with geographic distribution"
            })
        
        # Add cost estimates if requested
        if include_costs:
            # Rough AWS cost estimates (USD/month)
            ec2_cost = (base_cpu_cores * 0.0464 + base_ram_gb * 0.0116) * 24 * 30  # t3 instances
            storage_cost = base_storage_gb * 0.10  # GP3 SSD
            
            if concurrent_users >= 500:
                load_balancer_cost = 22.50  # ALB
                rds_cost = 200 + (base_ram_gb * 5)  # RDS PostgreSQL
            else:
                load_balancer_cost = 0
                rds_cost = 100  # Small RDS instance
            
            recommendations["cost_estimates"] = {
                "monthly_usd": {
                    "compute": round(ec2_cost, 2),
                    "storage": round(storage_cost, 2),
                    "load_balancer": round(load_balancer_cost, 2),
                    "database": round(rds_cost, 2),
                    "total_estimated": round(ec2_cost + storage_cost + load_balancer_cost + rds_cost, 2)
                },
                "disclaimer": "AWS estimates for reference only. Actual costs vary by region, usage patterns, and provider."
            }
        
        increment_counter("api_requests_total", {"endpoint": "infrastructure-requirements"})
        
        return recommendations
    
    except Exception as e:
        logger.error(f"Infrastructure requirements calculation failed: {e}")
        increment_counter("api_errors_total", {"endpoint": "infrastructure-requirements"})
        raise HTTPException(status_code=500, detail=f"Requirements calculation failed: {str(e)}")