# -*- coding: utf-8 -*-

"""
Integration tests for MT5 client and server components.
This test runs both the client and server tests.
"""

import sys
import subprocess
from pathlib import Path
from rich.console import Console

# Add parent directory to path to import qlib modules
parent_dir = str(Path(__file__).resolve().parent.parent.parent.parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

console = Console()

def run_test(test_file):
    """Run a test file and return result"""
    console.print(f"[bold blue]Running test: {test_file.name}[/bold blue]")
    
    # Run the test file
    result = subprocess.run(
        [sys.executable, str(test_file)],
        cwd=parent_dir,
        capture_output=True,
        text=True
    )
    
    # Print output
    if result.stdout:
        console.print(result.stdout)
    
    # Print error if any
    if result.returncode != 0:
        console.print(f"[bold red]Test failed: {test_file.name}[/bold red]")
        if result.stderr:
            console.print(f"[red]{result.stderr}[/red]")
        return False
    
    console.print(f"[bold green]Test passed: {test_file.name}[/bold green]")
    return True

if __name__ == "__main__":
    console.print("[bold blue]Running MT5 Integration Tests[/bold blue]")
    
    # Get the directory of this file
    current_dir = Path(__file__).resolve().parent.parent
    
    # Get all test files
    test_files = [
        current_dir / "test_mt5_client.py",
        current_dir / "test_mt5_server.py"
    ]
    
    # Run all tests
    all_passed = True
    for test_file in test_files:
        if not test_file.exists():
            console.print(f"[bold yellow]Warning: Test file not found: {test_file.name}[/bold yellow]")
            all_passed = False
            continue
        
        if not run_test(test_file):
            all_passed = False
    
    # Print summary
    if all_passed:
        console.print("[bold green]All tests passed![/bold green]")
        sys.exit(0)
    else:
        console.print("[bold red]Some tests failed![/bold red]")
        sys.exit(1) 
#!/usr/bin/env python3
 