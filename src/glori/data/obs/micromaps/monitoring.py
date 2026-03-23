import os
import sys
import threading
import time
import traceback
import datetime
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    import psutil

    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

# local utils used in your project (keep as in original file)
import glori.settings.paths as paths
from glori.infra.logging import get_logger

logger = get_logger("monitoring")

# monitor state
_MONITOR_THREAD = None
_MONITOR_STOP = None
_MONITOR_LOGPATH = None
# no in-memory lock/state needed when using append-only logfile


def _parse_monitor_args(argv):
    enabled = any(a == "--monitor" or a.startswith("--monitor") for a in argv)
    interval = 5.0
    show_paths = False
    print_log = False
    out_dir = None
    plot_interval = 60.0
    for a in argv:
        if a.startswith("--monitor-interval="):
            try:
                interval = float(a.split("=", 1)[1])
            except Exception:
                pass
        if a == "--monitor-show-paths":
            show_paths = True
        if a == "--monitor-print-log":
            print_log = True
        if a.startswith("--monitor-out="):
            out_dir = a.split("=", 1)[1]
        if a.startswith("--monitor-plot-interval="):
            try:
                plot_interval = float(a.split("=", 1)[1])
            except Exception:
                pass
    return enabled, interval, show_paths, out_dir, plot_interval, print_log


def _gather_process_tree_stats(pid):
    """
    Return (num_fds, rss_bytes, cpu_time_seconds) aggregated over pid + children when possible.
    Falls back to /proc for systems without psutil.
    """
    if _HAS_PSUTIL:
        try:
            proc = psutil.Process(pid)
            # num fds (proc + children)
            try:
                num_fds = proc.num_fds()
            except Exception:
                num_fds = len(os.listdir(f"/proc/{pid}/fd"))
            rss = 0
            cpu = 0.0
            try:
                mi = proc.memory_info()
                rss += getattr(mi, "rss", 0)
            except Exception:
                pass
            try:
                ct = proc.cpu_times()
                cpu += getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0)
            except Exception:
                pass
            try:
                for child in proc.children(recursive=True):
                    try:
                        mi = child.memory_info()
                        rss += getattr(mi, "rss", 0)
                    except Exception:
                        pass
                    try:
                        ct = child.cpu_times()
                        cpu += getattr(ct, "user", 0.0) + getattr(ct, "system", 0.0)
                    except Exception:
                        pass
                    try:
                        num_fds += child.num_fds()
                    except Exception:
                        pass
            except Exception:
                pass
            return int(num_fds), int(rss), float(cpu)
        except Exception:
            # fallthrough to /proc-based fallback
            pass

    # fallback: /proc for the single pid (no children)
    try:
        num_fds = len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        num_fds = 0
    rss = 0
    try:
        with open(f"/proc/{pid}/status", "r") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    parts = line.split()
                    if len(parts) >= 2:
                        rss = int(parts[1]) * 1024
                        break
    except Exception:
        pass
    cpu = 0.0
    try:
        with open(f"/proc/{pid}/stat", "r") as fh:
            parts = fh.read().split()
            if len(parts) > 15:
                utime = float(parts[13])
                stime = float(parts[14])
                clk = os.sysconf(os.sysconf_names["SC_CLK_TCK"])
                cpu = (utime + stime) / clk
    except Exception:
        pass
    return int(num_fds), int(rss), float(cpu)


def _save_monitor_plot(data, out_dir=None, fname_prefix="monitor"):
    """
    Save PNG plotting open fds and RSS (MB) vs walltime (bottom x-axis).
    If `data` is None, read the append-only CSV log file (created by the monitor thread).
    """
    if data is None:
        # read from logfile if present
        logpath = _MONITOR_LOGPATH
        if logpath is None or not Path(logpath).exists():
            return
        arr = np.genfromtxt(
            str(logpath), delimiter=",", names=True, dtype=None, encoding=None
        )
        if arr.size == 0:
            return
        # ensure arrays for single-line files
        t_wall = np.atleast_1d(arr["t_wall"]).astype(float)
        t_cpu = np.atleast_1d(arr["t_cpu"]).astype(float)
        fds = np.atleast_1d(arr["fds"]).astype(float)
        rss_mb = np.atleast_1d(arr["rss_bytes"]).astype(float) / (1024.0**2)
        pid = Path(logpath).stem.split("_")[-1]
        out_dir = Path(logpath).parent if out_dir is None else Path(out_dir)
    else:
        if out_dir is None:
            out_dir = paths.ANALYSIS_PARENT / "monitoring"
        else:
            out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        t_wall = np.array(data["t_wall"])
        t_cpu = np.array(data["t_cpu"])
        fds = np.array(data["fds"])
        rss_mb = np.array(data["rss_bytes"]) / (1024.0**2)
        pid = data.get("pid", "?")

    if len(t_wall) < 2:
        return

    fig, ax1 = plt.subplots(figsize=(10, 4))
    ax1.plot(t_wall - t_wall[0], fds, label="open fds", color="C0")
    ax1.set_xlabel("walltime (s)")
    ax1.set_ylabel("open fds", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    ax2 = ax1.twinx()
    ax2.plot(t_wall - t_wall[0], rss_mb, label="RSS (MB)", color="C1")
    ax2.set_ylabel("RSS (MB)", color="C1")
    ax2.tick_params(axis="y", labelcolor="C1")

    # top x-axis: Timestamps
    ax_top = ax1.twiny()
    ax_top.set_xlim(ax1.get_xlim())
    ax_top.set_xticks(ax1.get_xticks())
    time_labels = [
        (datetime.datetime.fromtimestamp(t_wall[0]) + datetime.timedelta(seconds=dt))
        for dt in ax1.get_xticks()
    ]
    time_strs = [tl.strftime("%H:%M:%S") for tl in time_labels]
    ax_top.set_xticklabels(time_strs)
    ax_top.set_xlabel("time (HH:MM:SS)")

    plt.title(f"Resource monitor (pid={pid})")
    plt.tight_layout()
    out_path = out_dir / f"{fname_prefix}.png"
    try:
        fig.savefig(out_path, dpi=150)
    except Exception as e:
        logger.warning(f"Failed to save monitor plot: {e}")
        traceback.print_exc()
    plt.close(fig)


def _monitor_thread_func(
    stop_event,
    pid=None,
    interval=5.0,
    show_paths=False,
    out_dir=None,
    plot_interval=60.0,
    logger=None,
    print_log=False,
):
    pid = pid or os.getpid()
    start_wall = time.time()
    start_wall_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    last_plot = start_wall

    # prepare logfile path
    global _MONITOR_LOGPATH
    if out_dir is None:
        out_dir = paths.ANALYSIS_PARENT / "monitoring"
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logpath = Path(out_dir) / f"monitor_{start_wall_str}.csv"
    _MONITOR_LOGPATH = str(logpath)
    # write header if file doesn't exist
    if not logpath.exists():
        with open(logpath, "w") as fh:
            fh.write("t_wall,t_cpu,fds,rss_bytes\n")

    while not stop_event.is_set():
        try:
            now = time.time()
            rel_wall = now - start_wall
            num_fds, rss, cpu = _gather_process_tree_stats(pid)

            # append sample to CSV logfile (single writer -> no lock)
            try:
                with open(logpath, "a") as fh:
                    fh.write(f"{now},{cpu},{num_fds},{rss}\n")
                    fh.flush()
                    # optional durability: os.fsync(fh.fileno())
            except Exception:
                # if file append fails, just continue (monitor should not crash)
                pass

            # optional logging of a summary line (no lock held here)
            if print_log:
                if logger is not None:
                    logger.info(
                        f"[monitor] pid={pid} wall={rel_wall:.1f}s cpu={cpu:.2f}s fds={num_fds} rss={rss//1024}kB"
                    )
                else:
                    print(
                        f"[monitor] pid={pid} wall={rel_wall:.1f}s cpu={cpu:.2f}s fds={num_fds} rss={rss//1024}kB"
                    )

            # optional: show a few open paths
            if show_paths:
                try:
                    if _HAS_PSUTIL:
                        proc = psutil.Process(pid)
                        paths_list = [
                            getattr(f, "path", "") for f in proc.open_files()
                        ][:10]
                        if print_log:
                            if logger:
                                logger.info("open paths: " + ",".join(paths_list))
                            else:
                                print("open paths:", paths_list)
                    else:
                        fd_dir = f"/proc/{pid}/fd"
                        entries = os.listdir(fd_dir)[:10]
                        targets = []
                        for fd in entries:
                            try:
                                targets.append(os.readlink(os.path.join(fd_dir, fd)))
                            except Exception:
                                targets.append("<unk>")
                        if print_log:
                            if logger:
                                logger.info("open paths: " + ",".join(targets))
                            else:
                                print("open paths:", targets)
                except Exception:
                    pass

            # trigger periodic plot: read from logfile (no lock) and save
            if (now - last_plot) >= plot_interval:
                try:
                    if print_log:
                        if logger:
                            logger.info(
                                f"Saving monitor plot to {out_dir} (from {logpath})"
                            )
                        else:
                            print(f"Saving monitor plot to {out_dir} (from {logpath})")
                    _save_monitor_plot(
                        None, out_dir=out_dir, fname_prefix=f"monitor-{start_wall_str}"
                    )
                except Exception as e:
                    if logger:
                        logger.warning(f"plot save failed: {e}")
                    traceback.print_exc()
                last_plot = now

        except Exception as e:
            if logger:
                logger.debug(f"monitor exception: {e}")
            else:
                print("monitor exception:", e)
        # wait with timeout so stop_event can interrupt sooner
        stop_event.wait(interval)

    # final plot from logfile
    try:
        # final plot from logfile
        if out_dir is not None:
            _save_monitor_plot(None, out_dir=out_dir, fname_prefix="monitor_final")
    except Exception:
        pass


def start_monitor(
    pid=None,
    interval=5.0,
    show_paths=False,
    out_dir=None,
    plot_interval=60.0,
    logger=None,
    print_log=False,
):
    """
    Start background monitoring thread. Idempotent.
    Returns (thread, stop_event).
    """
    global _MONITOR_THREAD, _MONITOR_STOP
    if _MONITOR_THREAD is not None and _MONITOR_THREAD.is_alive():
        return _MONITOR_THREAD, _MONITOR_STOP

    stop_event = threading.Event()
    thread = threading.Thread(
        target=_monitor_thread_func,
        args=(
            stop_event,
            pid,
            interval,
            show_paths,
            out_dir,
            plot_interval,
            logger,
            print_log,
        ),
        daemon=True,
        name="resource-monitor",
    )
    _MONITOR_THREAD = thread
    _MONITOR_STOP = stop_event
    thread.start()
    return thread, stop_event


def stop_monitor(wait=True):
    """
    Stop the background monitor and block until it finishes if wait=True.
    """
    global _MONITOR_THREAD, _MONITOR_STOP
    if _MONITOR_STOP is None:
        return
    _MONITOR_STOP.set()
    if wait and _MONITOR_THREAD is not None:
        _MONITOR_THREAD.join(timeout=5.0)
    _MONITOR_THREAD = None
    _MONITOR_STOP = None


import functools


def monitor(
    pid=None,
    interval=5.0,
    show_paths=False,
    out_dir=None,
    plot_interval=60.0,
    logger=None,
    parse_args=False,
):
    """
    Decorator that starts the background monitor before the wrapped function runs
    and stops it after the function returns (or on exception).

    Usage:
      @monitor(parse_args=True)   # reads CLI monitor flags via _parse_monitor_args(sys.argv)
      def main(...): ...
    or
      @monitor(interval=2.0, out_dir="/tmp/mon")
      def main(...): ...
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if parse_args:
                enabled, i, sp, out, plot_i, print_log = _parse_monitor_args(sys.argv)
                if not enabled:
                    return func(*args, **kwargs)
                th, stop = start_monitor(
                    pid=pid,
                    interval=i,
                    show_paths=sp,
                    out_dir=out,
                    plot_interval=plot_i,
                    logger=logger,
                    print_log=print_log,
                )
            else:
                th, stop = start_monitor(
                    pid=pid,
                    interval=interval,
                    show_paths=show_paths,
                    out_dir=out_dir,
                    plot_interval=plot_interval,
                    logger=logger,
                    print_log=False,
                )
            try:
                return func(*args, **kwargs)
            finally:
                stop_monitor(wait=True)

        return wrapper

    return decorator


# module CLI entrypoint: allow running the monitor standalone
if __name__ == "__main__":
    (
        _monitor_enabled,
        _monitor_interval,
        _monitor_show_paths,
        _monitor_out_dir,
        _monitor_plot_interval,
        _monitor_print_log,
    ) = _parse_monitor_args(sys.argv)
    if _monitor_enabled:
        start_monitor(
            pid=os.getpid(),
            interval=_monitor_interval,
            show_paths=_monitor_show_paths,
            out_dir=_monitor_out_dir,
            plot_interval=_monitor_plot_interval,
            logger=logger,
            print_log=_monitor_print_log,
        )
    # keep main thread alive so the monitor can run until interrupted
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        stop_monitor(wait=True)
