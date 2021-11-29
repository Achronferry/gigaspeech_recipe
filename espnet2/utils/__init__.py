"""Initialize sub package."""
def setup_logging_config(verbose, prefix="", color=True):
    import logging
    if prefix != "":
        format=f"{prefix} "
    else:
        format=""

    if color:
        from colorlog.logging import basicConfig
        # from espnet.asr.asr_utils import logging_once
        format+=f"%(log_color)s%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s%(reset)s"
        log_colors={
                    'DEBUG':    'cyan',
                    'INFO':     'white',
                    'WARNING':  'yellow',
                    'ERROR':    'red',
                    'CRITICAL': 'red,bg_white',
                }
        basicConfig(
            level=logging.INFO,
            format=format,
            log_colors=log_colors
        )
    else:
        format+=f"%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
        logging.basicConfig(
            level=logging.INFO,
            format=format,
        )


    # logging info
    if verbose == 0 or verbose == "NOTICE":
        logging.getLogger().setLevel(logging.INFO)
    elif verbose == 1 or verbose == "INFO":
        logging.getLogger().setLevel(logging.INFO)
    elif verbose == 2 or verbose == "DEBUG":
        logging.getLogger().setLevel(logging.DEBUG)
    elif verbose == -1 or verbose == "WARN" or verbose == "WARNING":
        logging.getLogger().setLevel(logging.WARN)
    elif verbose == -2 or verbose == "ERROR":
        logging.getLogger().setLevel(logging.ERROR)
    elif verbose == -3 or verbose == "CRITICAL":
        logging.getLogger().setLevel(logging.CRITICAL)
    else:
        logging.getLogger().setLevel(logging.INFO)
