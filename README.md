# Methanol Test Setup – Data Acquisition & Dashboard

This repository contains the full data pipeline for the **methanol / N₂ test setup**:

- Reads **sensor channels** via an **IFM IO-Link master**
- Streams values to a **Dash web dashboard**
- Logs experiments to **CSV** for offline analysis
- Can run against **real hardware** or a **simulation** (for offline development)

## 1. System Architecture

For a visual overview, see:

- **Hardware & wiring**  
  [System Architecture REV00 (PDF)](Documents/System%20Architecture%20REV00.pdf)
  (Sensor connections to IO board / IO-Link master)

- **Software & data flow**  
  [Architecture Diagram (PDF)](Documents/architecture.drawio.pdf) 
  (IO-Link master → Python → Acquisition → Dashboard & Logging)

These diagrams are the canonical references for wiring and software architecture.

## 2. Hardware Overview

### 2.1 IO-Link master & network

- **IO-Link master**: IFM AL1322 (8-port IO-Link master with Ethernet)
- **Connection to PC**: Ethernet (RJ45), using a **static IP**
- **Python** talks to the IO-Link master via **HTTP/JSON** (REST API)

Configure the IP in `config.py`:

```python
# IP address of the IO-Link master (AL1322)
IOT_HOST = "192.168.1.250"  # change when needed
AUTH_B64 = None             # e.g. "Basic abcdef..." if auth is enabled
```

## 3. Run code

1. Clone repository.
2. Connect hardware to laptop. (IO-link --> ethernet, IP-adress, Connect through Moneo)
3. Activate venv.
4. Check sensor configuration in config,py
5. run code with `python -m methanol_dashboard.run`
6. open `http://localhost:8050/` in browser.

## 4. Current port layout

```python
PORT_SD8500             = 1              # Testo SD8500
PORT_SD6500_1           = 2              # Testo SD6500 #1
PORT_SD6500_2           = 3              # Testo SD6500 #2
PORT_SENXTX_ANALOG      = 4              # SenxTx via analogue IO-Link (DP2200 or similar)
PORT_MICHELL_ANALOG     = 5              # Dewpoint Michell via analogue IO-Link
PORT_BANNER_DEWPOINT_1  = 6              # Dewpoint Banner #1 via Modbus–IO–Link converter
PORT_BANNER_DEWPOINT_2  = 7              # Dewpoint Banner #2 via Modbus–IO–Link converter
PORT_PT100_MODULE       = 8              # 4-channel PT100 IO-Link module (AL2284)
```