#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Script to check the health of migrated FX data using QLib's data health checker.
This script will:
1. Run health checks on the FX data
2. Display detailed reports for any issues found
3. Provide guidance for fixing common problems
"""

import os
import sys
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Add qlib parent directory to system path
repo_dir = Path(__file__).resolve().parent.parent.parent.parent.parent
if str(repo_dir) not in sys.path:
    sys.path.append(str(repo_dir))

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config import DEFAULT_FX_DATA_PATH

from scripts.check_data_health import DataHealthChecker

console = Console()

def check_m5_data_health(qlib_dir, large_step_threshold_price=0.05):
    """
    Check data health for M5 FX data
    
    Args:
        qlib_dir: QLib data directory
        large_step_threshold_price: Threshold for detecting large price changes 
                                   (lower for FX since they move less than stocks)
    """
    console.print(f"\n[bold cyan]Checking data health for M5 FX data[/bold cyan]")
    
    try:
        # Initialize the data health checker
        checker = DataHealthChecker(
            qlib_dir=qlib_dir,
            freq="5t",
            large_step_threshold_price=large_step_threshold_price,
            large_step_threshold_volume=0,  # We don't care about volume for FX
            missing_data_num=0
        )
        
        # Run the checks
        checker.check_data()
        
        # Store the results
        results = {
            "missing_data": checker.check_missing_data(),
            "large_steps": checker.check_large_step_changes(),
            "missing_columns": checker.check_required_columns(),
            "missing_factor": checker.check_missing_factor()
        }
        
    except Exception as e:
        console.print(f"[bold red]Error checking M5 data: {str(e)}[/bold red]")
        results = {"error": str(e)}
    
    return results

def generate_summary_report(results):
    """Generate a summary report of health check results"""
    console.print("\n[bold green]===== FX M5 DATA HEALTH CHECK SUMMARY =====[/bold green]")
    
    table = Table(title="Health Check Results")
    table.add_column("Check", style="cyan")
    table.add_column("Status", style="yellow")
    table.add_column("Details", style="magenta")
    
    if "error" in results:
        table.add_row(
            "Overall", 
            "❌ ERROR", 
            results["error"]
        )
    else:
        # Missing data
        status = "✅ OK" if results["missing_data"] is None else f"❌ Issues: {len(results['missing_data'])}"
        details = "" if results["missing_data"] is None else "Missing data points detected"
        table.add_row("Missing Data", status, details)
        
        # Large steps
        status = "✅ OK" if results["large_steps"] is None else f"❌ Issues: {len(results['large_steps'])}"
        details = "" if results["large_steps"] is None else "Large price jumps detected"
        table.add_row("Price Jumps", status, details)
        
        # Missing columns
        status = "✅ OK" if results["missing_columns"] is None else f"❌ Issues: {len(results['missing_columns'])}"
        details = "" if results["missing_columns"] is None else f"Missing: {results['missing_columns']}"
        table.add_row("Required Columns", status, details)
        
        # Missing factor
        status = "✅ OK" if results["missing_factor"] is None else f"❌ Issues: {len(results['missing_factor'])}"
        details = "" if results["missing_factor"] is None else "Factor column missing"
        table.add_row("Adjustment Factor", status, details)
    
    console.print(table)

def provide_recommendations(results):
    """Provide recommendations for fixing any issues found"""
    has_issues = False
    
    if "error" in results:
        has_issues = True
    elif any([
        results["missing_data"] is not None,
        results["large_steps"] is not None,
        results["missing_columns"] is not None,
        results["missing_factor"] is not None
    ]):
        has_issues = True
    
    if not has_issues:
        console.print(Panel("[bold green]No issues detected! Your FX M5 data looks healthy.[/bold green]", 
                           title="Recommendation", 
                           border_style="green"))
        return
    
    # Provide specific recommendations
    recommendations = []
    
    # Check for missing data
    if results.get("missing_data") is not None:
        recommendations.append(
            "- For missing data: Re-download the missing periods from the data source or "
            "interpolate using neighboring values if gaps are small."
        )
    
    # Check for large step changes
    if results.get("large_steps") is not None:
        recommendations.append(
            "- For large price jumps: Verify against another data source. For FX data, unexpected "
            "large jumps often indicate data quality issues or incorrect handling of weekends/holidays."
        )
    
    # Check for missing columns
    if results.get("missing_columns") is not None:
        recommendations.append(
            "- For missing columns: Ensure your migration script includes all required fields "
            "(open, high, low, close, volume). For FX data, use volume=1 as required."
        )
    
    # Check for missing factor
    if results.get("missing_factor") is not None:
        recommendations.append(
            "- For missing factor: FX data typically doesn't need adjustment factors, but QLib "
            "requires this field. Set factor=1.0 for all FX data points."
        )
    
    recommendation_text = "\n".join(recommendations)
    console.print(Panel(f"[bold yellow]Some issues were detected. Here are recommendations for fixing them:[/bold yellow]\n\n{recommendation_text}", 
                       title="Recommendations", 
                       border_style="yellow"))

def main():
    """Main function to run the FX data health check"""
    qlib_dir = os.path.expanduser(DEFAULT_FX_DATA_PATH)
    
    console.print(f"[bold]Running health checks on FX M5 data in {qlib_dir}[/bold]")
    
    # Adjust thresholds specifically for FX data
    # FX typically has smaller price movements than stocks
    results = check_m5_data_health(
        qlib_dir=qlib_dir,
        large_step_threshold_price=0.03  # 3% change for FX is already quite large
    )
    
    # Generate summary report
    generate_summary_report(results)
    
    # Provide recommendations
    provide_recommendations(results)
    
    console.print("\n[bold cyan]To use this FX M5 data in QLib:[/bold cyan]")
    console.print(f"  qlib.init(provider_uri='{qlib_dir}')")

if __name__ == "__main__":
    main() 