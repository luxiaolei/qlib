#!/usr/bin/env python
"""Debug script for testing Redis connectivity."""

import asyncio
import os
import sys

import redis.asyncio as redis
from rich.console import Console

console = Console()

async def test_redis_connection(redis_url):
    """Test Redis connection."""
    console.print(f"Testing Redis connection to: {redis_url}")
    
    try:
        # Create Redis client
        redis_client = redis.from_url(redis_url)
        
        # Test connection
        await redis_client.ping()
        console.print("[green]Successfully connected to Redis![/green]")
        
        # Test writing and reading
        test_key = "fmp_test_key"
        test_value = "test_value"
        await redis_client.set(test_key, test_value)
        retrieved = await redis_client.get(test_key)
        
        if retrieved.decode() == test_value:
            console.print(f"[green]Successfully wrote and read from Redis: {test_key} = {test_value}[/green]")
        else:
            console.print(f"[red]Value mismatch: {retrieved.decode()} != {test_value}[/red]")
            
        # Clean up
        await redis_client.delete(test_key)
        console.print("[green]Test key deleted[/green]")
        
        return True
        
    except Exception as e:
        console.print(f"[bold red]Redis connection error: {str(e)}[/bold red]")
        return False
        
async def main():
    """Main function."""
    # Use Redis URL from environment or default
    redis_url = os.environ.get("REDIS_URL", "redis://:EAFramwork2024@localhost:6379/0")
    
    console.print(f"[bold]FMP Data Collector Redis Debug[/bold]")
    
    # Test Redis connection
    success = await test_redis_connection(redis_url)
    
    if success:
        console.print("\n[bold green]All Redis tests passed![/bold green]")
        return 0
    else:
        console.print("\n[bold red]Redis tests failed![/bold red]")
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(main())) 