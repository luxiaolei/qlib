#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
MT5 QLib Data Service Installer

This script installs the MT5 client and server as systemd services.
It should be run on the systems where you want to run the services.

Usage:
    # On the MT5 server (VPS):
    python install_mt5_service.py --server --user mt5user
    
    # On the QLib system:
    python install_mt5_service.py --client --user xlmini --qlib_dir ~/.qlib/qlib_data/forex_data --server_url 192.168.160.1:8765
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from rich.console import Console

console = Console()

# Templates for systemd service files
MT5_SERVER_SERVICE = """[Unit]
Description=MT5 WebSocket Server
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={work_dir}
ExecStart={python} {script_path} --host {host} --port {port} --log_level {log_level}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""

MT5_CLIENT_SERVICE = """[Unit]
Description=MT5 QLib Client
After=network.target

[Service]
Type=simple
User={user}
WorkingDirectory={work_dir}
ExecStart={python} {script_path} --qlib_dir {qlib_dir} --server_url {server_url} --log_level {log_level}
Restart=on-failure
RestartSec=5
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
"""
# FIXME: the mt5 server is in windows!
def install_server_service(args):
    """Install the MT5 server systemd service"""
    try:
        # Get absolute path to script
        script_path = Path(__file__).resolve().parent / "mt5_server.py"
        work_dir = script_path.parent
        
        # Make sure script is executable
        os.chmod(script_path, 0o755)
        
        # Create service file
        service_content = MT5_SERVER_SERVICE.format(
            user=args.user,
            work_dir=work_dir,
            python=sys.executable,
            script_path=script_path,
            host=args.host,
            port=args.port,
            log_level=args.log_level
        )
        
        # Write service file
        service_file = Path(f"/tmp/mt5-server.service")
        with open(service_file, "w") as f:
            f.write(service_content)
        
        # Install service
        console.print(f"[bold]Installing MT5 server service...[/bold]")
        subprocess.run(["sudo", "mv", service_file, "/etc/systemd/system/"], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "mt5-server.service"], check=True)
        
        console.print("[green]MT5 server service installed successfully![/green]")
        console.print("\nTo start the service, run:")
        console.print("[bold]sudo systemctl start mt5-server.service[/bold]")
        
    except Exception as e:
        console.print(f"[red]Error installing server service: {e}[/red]")
        return False
    
    return True

def install_client_service(args):
    """Install the MT5 client systemd service"""
    try:
        # Get absolute path to script
        script_path = Path(__file__).resolve().parent / "mt5_qlib_client.py"
        work_dir = script_path.parent
        
        # Make sure script is executable
        os.chmod(script_path, 0o755)
        
        # Create service file
        service_content = MT5_CLIENT_SERVICE.format(
            user=args.user,
            work_dir=work_dir,
            python=sys.executable,
            script_path=script_path,
            qlib_dir=args.qlib_dir,
            server_url=args.server_url,
            log_level=args.log_level
        )
        
        # Write service file
        service_file = Path(f"/tmp/mt5-client.service")
        with open(service_file, "w") as f:
            f.write(service_content)
        
        # Install service
        console.print(f"[bold]Installing MT5 client service...[/bold]")
        subprocess.run(["sudo", "mv", service_file, "/etc/systemd/system/"], check=True)
        subprocess.run(["sudo", "systemctl", "daemon-reload"], check=True)
        subprocess.run(["sudo", "systemctl", "enable", "mt5-client.service"], check=True)
        
        console.print("[green]MT5 client service installed successfully![/green]")
        console.print("\nTo start the service, run:")
        console.print("[bold]sudo systemctl start mt5-client.service[/bold]")
        
    except Exception as e:
        console.print(f"[red]Error installing client service: {e}[/red]")
        return False
    
    return True

# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MT5 QLib Data Service Installer")
    
    # Mode selection arguments
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--server", action="store_true", help="Install the MT5 server service")
    mode_group.add_argument("--client", action="store_true", help="Install the MT5 client service")
    
    # Common arguments
    parser.add_argument("--user", type=str, default=os.environ.get("USER", "root"),
                      help="User to run the service as")
    parser.add_argument("--log_level", type=str, default="INFO",
                      choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                      help="Logging level")
    
    # Server arguments
    parser.add_argument("--host", type=str, default="0.0.0.0",
                      help="Host to bind the WebSocket server to (server only)")
    parser.add_argument("--port", type=int, default=8765,
                      help="Port to bind the WebSocket server to (server only)")
    
    # Client arguments
    parser.add_argument("--qlib_dir", type=str, default="~/.qlib/qlib_data/forex_data",
                      help="Directory where QLib data is stored (client only)")
    parser.add_argument("--server_url", type=str, default="192.168.160.1:8765",
                      help="URL of the MT5 WebSocket server (format: host:port) (client only)")
    
    args = parser.parse_args()
    
    # Install dependencies using uv
    console.print("[bold]Installing dependencies...[/bold]")
    if args.server:
        subprocess.run(["uv", "pip", "install", "websockets", "pandas", "numpy", "rich", "MetaTrader5"], check=True)
    else:
        subprocess.run(["uv", "pip", "install", "websockets", "pandas", "numpy", "rich"], check=True)
    
    # Install service
    if args.server:
        if not install_server_service(args):
            console.print("[red]Failed to install server service.[/red]")
            sys.exit(1)
    else:
        if not install_client_service(args):
            console.print("[red]Failed to install client service.[/red]")
            sys.exit(1)
    
    # Done
    console.print("[bold green]Installation completed successfully![/bold green]") 