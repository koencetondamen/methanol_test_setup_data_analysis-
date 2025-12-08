# Methanol Test Setup – Data Acquisition & Dashboard

This repository contains the full data pipeline for the **methanol / N₂ test setup**:

- Reads **11 sensor channels** via an **IFM IO-Link master**
- Streams values to a **Dash web dashboard**
- Logs experiments to **CSV** for offline analysis
- Can run against **real hardware** or a **simulation** (for offline development)

---

## 1. System Architecture

For a visual overview, see:

- **Hardware & wiring**  
  `Documents//System Architecture REV00.pdf`  
  (Sensor connections to IO board / IO-Link master)

- **Software & data flow**  
  `Documents//architecture.drawio.pdf`  
  (IO-Link master → Python → Acquisition → Dashboard & Logging)

These diagrams are the canonical references for wiring and software architecture.

---

## 2. Hardware Overview

### 2.1 IO-Link master & network

- **IO-Link master**: IFM AL1322 (8-port IO-Link master with Ethernet)
- **Connection to PC**: Ethernet (RJ45), using a **static IP**
- **Python** talks to the IO-Link master via **HTTP/JSON** (REST API)

Configure the IP in `config.py`:

```python
# IP address of the IO-Link master (AL1322)
IOT_HOST = "192.168.1.132"  # change when needed
AUTH_B64 = None             # e.g. "Basic abcdef..." if auth is enabled
