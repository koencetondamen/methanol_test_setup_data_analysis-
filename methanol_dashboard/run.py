# Methanol_dashboard/run.py
from . import config

# Choose master implementation based on simulation flag
if config.USE_SIMULATION:
    from .simulation import SimulatedIoLinkMaster as MasterClass

else:
    from .io_link_master import IoLinkMaster as MasterClass

from .acquisition import AcquisitionManager
from .experiment_log import ExperimentLogger
from .dashboard.app import create_app

def main() -> None:

    # Instantiate IO-Link master connection. 
    io_master = MasterClass(
        host=config.IOT_HOST, # the IP adress of the IO link
    )

    experiment_logger = ExperimentLogger(base_dir=config.EXPERIMENT_DIR) # start experiment logger, also add location experiment data.

    # Add all the sensors that are actively measured
    sample_kwargs = dict(
        sd8500_port=config.PORT_SD8500,
        sd6500_1_port=config.PORT_SD6500_1,
        sd6500_2_port=config.PORT_SD6500_2,
        senxtx_port=config.PORT_SENXTX_ANALOG,
        michell_port=config.PORT_MICHELL_ANALOG,
        banner_dp1_port=config.PORT_BANNER_DEWPOINT_1,
        banner_dp2_port=config.PORT_BANNER_DEWPOINT_2,
        banner_dp3_port=config.PORT_BANNER_DEWPOINT_3,
        pt100_module_port=config.PORT_PT100_MODULE,
        timeout=1.0,
    )

    acquisition = AcquisitionManager(
        io_master=io_master,
        sample_period_s=config.SAMPLE_PERIOD_S,
        history_seconds=config.HISTORY_MAX_SECONDS,
        sample_kwargs=sample_kwargs,
        experiment_logger=experiment_logger,
    )
    acquisition.start()

    app = create_app(acquisition, experiment_logger)
    app.run(host="0.0.0.0", port=8050, debug=True)

if __name__ == "__main__":
    main()
