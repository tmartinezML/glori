import types
import logging
import colorlog
import datetime
from pathlib import Path
from tqdm import tqdm


def separator(self, msg, level=logging.INFO, length=80):
    """
    Log a message with a separator line.

    Parameters
    ----------
    msg : str
        The message to log.
    level : int, optional
        The logging level, by default logging.INFO
    """
    log_divider(self, level=level, length=length, linebreak=(True, False))
    self.log(level, msg.rjust(length // 2 + len(msg) // 2, " ").ljust(length))
    log_divider(self, level=level, length=length, linebreak=(False, True))


def log_divider(self, level=logging.INFO, length=80, linebreak=(True, True), n_lines=1):

    for i in range(n_lines):
        if linebreak[0] and i == 0:
            print("\n")
        self.log(
            level,
            f"{'-' * length}" + ("\n" if linebreak[1] and i == n_lines - 1 else ""),
        )


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with the given name and level. If the logger already has
    handlers, clear and reset them.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : int, optional
        The logging level, by default logging.INFO
    enable_overwrite : bool, optional
        Whether to enable overwrite functionality, by default False

    Returns
    -------
    logging.Logger
        The logger with the given name and level.
    """
    logger = logging.getLogger(name)
    if logger.hasHandlers():  # Check if the logger already has handlers
        logger.handlers.clear()  # Clear the default handlers
    logger.setLevel(level)

    handler = logging.StreamHandler()

    # formatter = logging.Formatter(
    #     "%(asctime)s - %(levelname)s (%(name)s): %(message)s", "%H:%M:%S"
    # )
    # Create colored formatter
    formatter = colorlog.ColoredFormatter(
        "%(blue)s %(asctime)s (%(name)s)%(reset)s - %(log_color)s%(levelname)s %(reset)s: %(message)s",
        "%H:%M:%S",
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logger.divider = types.MethodType(log_divider, logger)
    logger.separator = types.MethodType(separator, logger)
    return logger


def add_file_handler(logger, log_file: str, level: int = None):
    """Add a file handler to an existing logger."""
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(log_file)
    file_formatter = colorlog.ColoredFormatter(
        "%(blue)s %(asctime)s (%(name)s)%(reset)s - %(log_color)s%(levelname)s %(reset)s: %(message)s",
        "%H:%M:%S",
        # "%(log_color)s%(levelname)-8s%(reset)s %(blue)s%(name)s%(reset)s: %(message)s",
        log_colors={
            "DEBUG": "cyan",
            "INFO": "green",
            "WARNING": "yellow",
            "ERROR": "red",
            "CRITICAL": "red,bg_white",
        },
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(level if level is not None else logger.level)
    logger.addHandler(file_handler)
    return logger


# ... rest of your existing functions remain unchanged ...
def pretty_print_config(config, title=None) -> None:
    """
    Pretty print a model configuration object.

    Args:
        config (modelConfig): The configuration object to be printed.
    """
    padding = 4
    indent_width = 2

    def print_recursive(d: dict, indent: int = 0, min_len: int = 0) -> None:
        if not len(d):
            print(" " * indent_width * indent + "{}")
            return
        max_key_length = max(min_len, max(len(k) for k in d.keys()))
        aligned_format = (
            " " * indent_width * (indent + 1)
            + f"{{:<{max_key_length + padding + indent_width * indent}}}: {{}}"
        )
        for k, v in d.items():
            if isinstance(v, dict):
                print(aligned_format.format(k, ""))
                print_recursive(v, indent + 1, min_len=0)
                print("")
            else:
                print(aligned_format.format(k, v))

    if isinstance(config, dict):
        param_dict = config
    elif isinstance(config, types.SimpleNamespace):
        param_dict = vars(config)
    else:

        assert hasattr(
            config, "param_dict"
        ), "Config object must have a 'param_dict' attribute."
        param_dict = config.param_dict

    if title is not None and len(title):
        print(f"\n{title}:\n")
    print_recursive(param_dict)
    print("\n")


pbar, last_loaded = None, 0


def show_dl_progress(block_num, block_size, total_size):
    """
    Designed as report_hook argument for urllib.request.urlretrieve. Displays
    a progress bar for the download.

    Parameters
    ----------
    block_num : float
        Number of blocks downloaded so far.
    block_size : float
        Size of blocks in bytes.
    total_size : float
        Total size of the download in bytes.

    Comments
    --------
    I didn't specifically check whether the arguments are actually float, so
    if your life depends on it, don't make your life depend on it.
    """
    global pbar, last_loaded
    if pbar is None:
        pbar = tqdm(total=total_size, unit="Bytes", unit_scale=True)

    downloaded = block_num * block_size
    increment = downloaded - last_loaded
    last_loaded = downloaded
    if downloaded < total_size:
        pbar.update(increment)
    else:
        pbar.close()
        pbar, last_loaded = None, 0


def format_timedelta(dt):
    if dt.microseconds % 1000 >= 500:  # check if there will be rounding up
        dt = dt + datetime.timedelta(milliseconds=1)  # manually round up
    return str(dt).split(".")[0]
