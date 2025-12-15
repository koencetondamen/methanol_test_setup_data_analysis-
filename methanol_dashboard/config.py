from pathlib import Path

# --------------------------
# IO-Link master configuration
# --------------------------

# Set this to True while the IO-Link master is not connected.
USE_SIMULATION = True

# IP address of the IO-Link master (AL1322)
IOT_HOST = "192.168.1.250"  # change when needed

# IO-Link port mapping on the 8-port master.
# Adjust these to match your final wiring once the hardware is connected.

PORT_SD8500             = None  # Testo SD8500
PORT_SD6500_1           = None  # Testo SD6500 #1
PORT_SD6500_2           = None  # Testo SD6500 #2
PORT_SENXTX_ANALOG      = None  # SenxTx via analogue IO-Link (DP2200 or similar)
PORT_MICHELL_ANALOG     = None  # Dewpoint Michell via analogue IO-Link
PORT_BANNER_DEWPOINT_1  = 6  # Dewpoint Banner #1 via Modbus–IO–Link converter
PORT_BANNER_DEWPOINT_2  = None  # Dewpoint Banner #2 via Modbus–IO–Link converter
PORT_PT100_MODULE       = 8  # 4-channel PT100 IO-Link module (AL2284)

# --------------------------
# Acquisition parameters
# --------------------------

SAMPLE_PERIOD_S = 1.0           # seconds between samples
HISTORY_MAX_SECONDS = 30 * 60   # keep 30 minutes of history in memory

# --------------------------
# Data / logging directories
# --------------------------

BASE_DATA_DIR = Path("data") # folder for the data
LOG_DIR = BASE_DATA_DIR / "logs" # file for logging
EXPERIMENT_DIR = BASE_DATA_DIR / "experiments" # file for experiments 

# --------------------------
# Dashboard sensor mapping
# --------------------------
# These are the 11 logical sensor signals that will appear on the dashboard.

SENSOR_FIELDS = [
    # {
    #     "field": "sd8500_flow_m3_h",
    #     "label": "SD8500 – flow [m³/h]",
    #     "unit": "m³/h",
    # },
    # {
    #     "field": "sd6500_1_flow_m3_h",
    #     "label": "SD6500 #1 – flow [m³/h]",
    #     "unit": "m³/h",
    # },
    # {
    #     "field": "sd6500_1_temperature_c",
    #     "label": "SD6500 #1 – temperature [°C]",
    #     "unit": "°C",
    # },
    # {
    #     "field": "sd6500_2_flow_m3_h",
    #     "label": "SD6500 #2 – flow [m³/h]",
    #     "unit": "m³/h",
    # },
    # {
    #     "field": "senxtx_o2_current_mA",
    #     "label": "SenxTx – current [mA]",
    #     "unit": "mA",
    # },
    # {
    #     "field": "senxtx_o2_oxygen_percent",
    #     "label": "SenxTx – level [%]",
    #     "unit": "%",
    # },
    # {
    #     "field": "dewpoint_michell_current_mA",
    #     "label": "Dewpoint Michell – current [mA]",
    #     "unit": "mA",
    # },
    {
        "field": "dewpoint_banner_1_Humidity",
        "label": "Dewpoint Banner #1 – Humidity [%]",
        "unit": "%",
    },
    {
        "field": "dewpoint_banner_1_degreeC",
        "label": "Dewpoint Banner #1 – Temperature [°C]",
        "unit": "°C",
    },
    {
        "field": "dewpoint_banner_1_dewpoint",
        "label": "Dewpoint Banner #1 – dewpoint [°C]",
        "unit": "°C",
    },
    # {
    #     "field": "dewpoint_banner_2_degC",
    #     "label": "Dewpoint Banner #2 – dewpoint [°C]",
    #     "unit": "°C",
    # },
    {
        "field": "pt100_1_degC",
        "label": "PT100 #1 – temperature [°C]",
        "unit": "°C",
    },
    # {
    #     "field": "pt100_2_degC",
    #     "label": "PT100 #2 – temperature [°C]",
    #     "unit": "°C",
    # },
    # {
    #     "field": "pt100_3_degC",
    #     "label": "PT100 #3 – temperature [°C]",
    #     "unit": "°C",
    # },
    # {
    #     "field": "pt100_4_degC",
    #     "label": "PT100 #4 – temperature [°C]",
    #     "unit": "°C",
    # },
]
