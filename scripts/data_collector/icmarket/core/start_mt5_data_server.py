# -*- coding: utf-8 -*-
"""
# MetaTrader 5 Data Server

This script starts a WebSocket server that provides real-time and historical data from MetaTrader 5 (MT5) to
connected clients. The server uses the `MT5Wrapper` class from the `mt5tools` package to interact with the MT5 terminal.

## Usage

To start the server, run the script with the following optional command-line arguments:

- `server_url`: The URL of the server in the format `host:port` (default: `192.168.160.1:8765`).
- `mt5_exe_path`: The path to the MetaTrader 5 executable (default: `C:\Program Files\MetaTrader 5\terminal64.exe`).

Example:

```
python server.py 192.168.160.1:8765 "C:\Program Files\MetaTrader 5\terminal64.exe"
```

## Client Example

Here's an example of how a client can connect to the server and receive data:

```python
import asyncio
import json
import websockets

async def client():
    async with websockets.connect('ws://192.168.160.1:8765') as websocket:
        # Send a subscription message to the server
        request = {
            "symbol": "EURUSD",
            "timeframe": "M1",
            "history": 1000
        }
        await websocket.send(json.dumps(request))

        # Receive and process data from the server
        while True:
            data = await websocket.recv()
            bars = json.loads(data)
            print(f"Received {len(bars)} bars")
            # Process the received bars

asyncio.get_event_loop().run_until_complete(client())
```

In this example, the client connects to the server at `ws://192.168.160.1:8765` and sends a subscription message
with the desired symbol, timeframe, and history length. The server responds with historical data and continues to
send updates at the specified timeframe interval. The client receives and processes the data as it arrives.

## Server Details

The server handles each client connection in a separate coroutine (`handle_client`), which receives the subscription
message, sends the initial historical data, and then starts sending incremental updates using the `send_updates` coroutine.

The `get_data` function retrieves the requested data from MT5 using the `MT5Wrapper` instance.

Note: Make sure to have the `mt5tools` package installed and the MetaTrader 5 terminal running with the specified
`mt5_exe_path` for the server to function properly.

## Data Format

The server sends data to the client in JSON format. Each message contains an array of bar objects, where each bar
object represents a single candlestick and has the following properties:

- `time`: The timestamp of the bar.
- `open`: The opening price of the bar.
- `high`: The highest price of the bar.
- `low`: The lowest price of the bar.
- `close`: The closing price of the bar.
- `tick_volume`: The tick volume of the bar.
- `spread`: The spread of the bar.
- `real_volume`: The real volume of the bar.

Example bar object:

```json
{
  "time": 1620120000,
  "open": 1.20165,
  "high": 1.20188,
  "low": 1.20084,
  "close": 1.20121,
  "tick_volume": 79,
  "spread": 10,
  "real_volume": 0
}
```

The client can expect to receive an initial message containing the requested historical data, followed by incremental
updates at the specified timeframe interval.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import websockets
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mt5tools.wrapper import MT5Wrapper


class DataRequest(BaseModel):
    symbol: str
    timeframe: str
    history: int


def get_data(
    mt5wrapper: MT5Wrapper, symbol: str, timeframe: str, lb: int
) -> List[Dict[str, Any]]:
    bars = mt5wrapper.get_bars(symbol, timeframe, look_back=lb, return_json=True)
    return bars  # type: ignore


async def send_updates(websocket, mt5wrapper: MT5Wrapper, symbol: str, timeframe: str):
    """
    Send the last bar every `timeframe` time interval.
    It sleeps for 15 seconds to avoid sending the same bar twice.
    It also checks if the last data is different from the current data.
    If so, it sends the current bar.
    """
    print(f"Start sending incremental updates...")
    last_data = []
    send_count = 0
    loop = asyncio.get_event_loop()
    while True:
        await asyncio.sleep(15)
        data = await loop.run_in_executor(
            None, get_data, mt5wrapper, symbol, timeframe, 2
        )
        if data == last_data or len(data) == 0:
            continue
        last_data = data
        await websocket.send(json.dumps(data))
        send_count += 1
        if send_count % 10 == 0:
            print(f"{symbol} sent {send_count} times")


async def handle_client(websocket, mt5wrapper: MT5Wrapper):
    print(f"New client connected")
    try:
        loop = asyncio.get_event_loop()
        # Receive subscription message from the client
        message = await websocket.recv()
        request = DataRequest.parse_raw(message)
        print(f"Received request: {request}")

        # Send historical data back to the client
        historical_data = await loop.run_in_executor(
            None,
            get_data,
            mt5wrapper,
            request.symbol,
            request.timeframe,
            request.history,
        )

        print(f"Sending initial historical data...")
        await websocket.send(json.dumps(historical_data))
        await asyncio.sleep(0)  # yield control to the event loop

        # Start sending data at the requested timeframe interval
        await send_updates(websocket, mt5wrapper, request.symbol, request.timeframe)
    except websockets.exceptions.ConnectionClosed as e:  # type: ignore
        print(f"Client connection closed: {e}")
    except Exception as e:
        print(f"Error while handling client: {e}")


def main(server_url: str, mt5_exe_path: str):
    try:
        # Attempt to fix path issues (if any)
        mt5_exe_path = Path(mt5_exe_path).resolve()  # type: ignore
        assert Path(mt5_exe_path).exists(), f"Invalid MT5 exe path: {mt5_exe_path}"
        print(f"Using MT5 exe path: {mt5_exe_path}")

        mt5wrapper = MT5Wrapper(str(mt5_exe_path))

        if ":" not in server_url:
            server_url = f"{server_url}:8765"

        host, port = server_url.split(":")
        start_server = websockets.serve(lambda websocket: handle_client(websocket, mt5wrapper), host, int(port), ping_timeout=None)  # type: ignore
        print(f"Starting server at {server_url}")
        asyncio.get_event_loop().run_until_complete(start_server)
        asyncio.get_event_loop().run_forever()
    except Exception as e:
        print(f"Error while running the server: {e}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the data server.")
    parser.add_argument(
        "server_url",
        nargs="?",
        default="192.168.160.1:8765",
        help="The server URL (default: 192.168.160.1:8765)",
    )
    parser.add_argument(
        "mt5_exe_path",
        nargs="?",
        default=r"C:\Program Files\MetaTrader 5\terminal64.exe",
        help="The path to the MetaTrader 5 executable (default: C:\Program Files\MetaTrader 5\terminal64.exe)",
    )  # type: ignore
    args = parser.parse_args()
    main(args.server_url, args.mt5_exe_path)
