from io_to_master_fotios import IoLinkMaster

IOT_HOST          = "192.168.1.250"
SD6500_PORT       = 1
SD6500_2_PORT     = 2
SD8500_PORT       = 3
TEMP_MODULE_PORT  = 8
DP2200_PORT       = 4

AUTH_B64         = None

csv_path = "sensor_log.csv"
period_s = 1.0

if __name__ == "__main__":

    sensors = IoLinkMaster(
        host=IOT_HOST,
        auth_b64=AUTH_B64,
    )

    sensors.log_loop_to_csv(
        csv_path          = csv_path,
        period_s          = period_s,
        sensors_to_log    = ["dp2200"], # ["sd6500", "sd6500_2", "sd8500", "al2284", "dp2200"],
        sd6500_port       = SD6500_PORT,
        sd6500_2_port     = SD6500_2_PORT,
        sd8500_port       = SD8500_PORT,
        temp_module_port  = TEMP_MODULE_PORT,
        dp2200_port       = DP2200_PORT,
        timeout           = 1.0,
    )
