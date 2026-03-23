#!/usr/bin/env python3
"""
RAM Usage Monitor - Track memory consumption over time.

Usage:
    python monitor_ram.py --interval 5 --duration 3600 --output ram_log.csv
    python monitor_ram.py --interval 1 --endless  # Run until Ctrl+C
"""

import argparse
import csv
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib.pyplot as plt
import psutil

import utils.paths as paths

OUT_PARENT = paths.ANALYSIS_PARENT / "monitor_ram"
OUT_PARENT.mkdir(parents=False, exist_ok=True)


def format_elapsed_time(seconds):
    """
    Format elapsed time as d-h-m-s, hiding zero components.

    Examples:
        90 -> "1m 30s"
        3661 -> "1h 1m 1s"
        90061 -> "1d 1h 1m 1s"
    """
    td = timedelta(seconds=int(seconds))

    days = td.days
    hours, remainder = divmod(td.seconds, 3600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days > 0:
        parts.append(f"{days}d")
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or len(parts) == 0:  # Always show seconds if everything else is 0
        parts.append(f"{secs}s")

    return " ".join(parts)


class RAMMonitor:
    """Monitor and log RAM usage over time."""

    def __init__(
        self,
        interval=5,
        duration=None,
        output_file=None,
        plot_every=60,
    ):
        """
        Initialize RAM monitor.

        Args:
            interval: Seconds between measurements
            duration: Total monitoring time in minutes (None for endless)
            output_file: Path to save CSV log
            plot_every: Seconds between plot updates (None to disable live plotting)
        """
        self.interval = interval
        self.duration = duration * 60 if duration is not None else None
        self.plot_every = plot_every

        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"ram_usage_{timestamp}.csv"
        self.output_file = OUT_PARENT / output_file
        self.plot_file = self.output_file.with_suffix(".png")

        # Data storage
        self.timestamps = []
        self.ram_usage_gb = []
        self.ram_percent = []

        # Track if we're running
        self.running = True

        # Track last plot time
        self.last_plot_time = 0

        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle Ctrl+C and other termination signals."""
        print(f"\n\n🛑 Received signal {signum}. Shutting down gracefully...")
        self.running = False

    def get_ram_usage(self):
        """Get current RAM usage."""
        mem = psutil.virtual_memory()
        return {
            "gb": mem.used / (1024**3),  # Convert to GB
            "percent": mem.percent,
            "total_gb": mem.total / (1024**3),
            "available_gb": mem.available / (1024**3),
        }

    def monitor(self):
        """Main monitoring loop."""
        print(f"📊 RAM Monitoring Started")
        print(f"   Interval: {self.interval}s")
        print(
            f"   Duration: {'Endless' if self.duration is None else format_elapsed_time(self.duration)}"
        )
        print(
            f"   Plot update: {'Disabled' if self.plot_every is None else f'Every {format_elapsed_time(self.plot_every)}'}"
        )
        print(f"   Output: {self.output_file}")
        print(f"   Plot: {self.plot_file}")
        print(f"\n🔍 Press Ctrl+C to stop monitoring\n")

        start_time = time.time()
        self.last_plot_time = start_time

        try:
            # Open CSV file for writing
            with open(self.output_file, "w", newline="") as csvfile:
                fieldnames = [
                    "timestamp",
                    "elapsed_seconds",
                    "ram_used_gb",
                    "ram_percent",
                    "ram_total_gb",
                    "ram_available_gb",
                ]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()

                iteration = 0
                while self.running:
                    current_time = time.time()

                    # Check if duration exceeded
                    elapsed = current_time - start_time
                    if self.duration is not None and elapsed > self.duration:
                        print(
                            f"\n⏰ Duration of {format_elapsed_time(self.duration)} reached. Stopping..."
                        )
                        break

                    # Get current RAM usage
                    ram = self.get_ram_usage()
                    timestamp = datetime.now()

                    # Store data
                    self.timestamps.append(timestamp)
                    self.ram_usage_gb.append(ram["gb"])
                    self.ram_percent.append(ram["percent"])

                    # Write to CSV
                    writer.writerow(
                        {
                            "timestamp": timestamp.isoformat(),
                            "elapsed_seconds": elapsed,
                            "ram_used_gb": ram["gb"],
                            "ram_percent": ram["percent"],
                            "ram_total_gb": ram["total_gb"],
                            "ram_available_gb": ram["available_gb"],
                        }
                    )
                    csvfile.flush()  # Ensure data is written

                    # Print status
                    if iteration % 10 == 0:  # Print header every 10 lines
                        print(
                            f"\n{'Time':<20} {'Elapsed':<15} {'Used (GB)':<12} {'Used (%)':<10} {'Available (GB)':<15}"
                        )
                        print("-" * 85)

                    print(
                        f"{timestamp.strftime('%H:%M:%S'):<20} "
                        f"{format_elapsed_time(elapsed):<15} "
                        f"{ram['gb']:>10.2f}  "
                        f"{ram['percent']:>8.1f}%  "
                        f"{ram['available_gb']:>13.2f}"
                    )

                    iteration += 1

                    # Update plot at regular intervals
                    if self.plot_every is not None:
                        time_since_last_plot = current_time - self.last_plot_time
                        if time_since_last_plot >= self.plot_every:
                            print(
                                f"\n📈 Updating plot... (elapsed: {format_elapsed_time(elapsed)})"
                            )
                            self._save_plot(show_message=False)
                            self.last_plot_time = current_time

                    # Sleep until next measurement
                    time.sleep(self.interval)

        except Exception as e:
            print(f"\n❌ Error during monitoring: {e}")
            import traceback

            traceback.print_exc()

        finally:
            # Always save final plot before exiting
            print(f"\n📈 Generating final plot...")
            self._save_plot(show_message=True)

    def _save_plot(self, show_message=True):
        """Save RAM usage plot."""
        if len(self.timestamps) == 0:
            if show_message:
                print("⚠️  No data collected. Skipping plot generation.")
            return

        # Calculate elapsed time in seconds from start
        start_time = self.timestamps[0]
        elapsed_seconds = [(t - start_time).total_seconds() for t in self.timestamps]

        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

        # Plot 1: RAM usage in GB
        ax1.plot(self.timestamps, self.ram_usage_gb, linewidth=2, color="#2E86AB")
        ax1.fill_between(self.timestamps, self.ram_usage_gb, alpha=0.3, color="#2E86AB")
        ax1.set_ylabel("RAM Used (GB)", fontsize=12, fontweight="bold")
        ax1.grid(True, alpha=0.3)
        ax1.set_title(
            f"RAM Usage Monitor - {self.timestamps[0].strftime('%Y-%m-%d %H:%M:%S')} (Live)",
            fontsize=14,
            fontweight="bold",
        )

        # Add statistics to plot 1
        mean_gb = sum(self.ram_usage_gb) / len(self.ram_usage_gb)
        max_gb = max(self.ram_usage_gb)
        min_gb = min(self.ram_usage_gb)
        ax1.axhline(
            mean_gb, color="green", linestyle="--", label=f"Mean: {mean_gb:.2f} GB"
        )
        ax1.axhline(max_gb, color="red", linestyle="--", label=f"Max: {max_gb:.2f} GB")
        ax1.legend(loc="upper right")

        # Create secondary x-axis for elapsed time (top of ax1)
        ax1_top = ax1.twiny()
        ax1_top.set_xlim(ax1.get_xlim())  # Match the main axis limits

        # Convert timestamp limits to elapsed time for the top axis
        import matplotlib.dates as mdates
        from matplotlib.ticker import FuncFormatter

        def elapsed_formatter(x, pos):
            """Format elapsed time as HH:MM"""
            # x is in matplotlib date format, convert to timestamp
            try:
                ts = mdates.num2date(x)
                elapsed = (ts - mdates.num2date(ax1.get_xlim()[0])).total_seconds()
                hours = int(elapsed // 3600)
                minutes = int((elapsed % 3600) // 60)
                return f"{hours:02d}:{minutes:02d}"
            except:
                return ""

        ax1_top.xaxis.set_major_formatter(FuncFormatter(elapsed_formatter))
        ax1_top.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax1_top.set_xlabel("Elapsed Time (HH:MM)", fontsize=11, fontweight="bold")

        # Plot 2: RAM usage in percent
        ax2.plot(self.timestamps, self.ram_percent, linewidth=2, color="#A23B72")
        ax2.fill_between(self.timestamps, self.ram_percent, alpha=0.3, color="#A23B72")
        ax2.set_xlabel("Time (HH:MM)", fontsize=12, fontweight="bold")
        ax2.set_ylabel("RAM Used (%)", fontsize=12, fontweight="bold")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)

        # Add statistics to plot 2
        mean_pct = sum(self.ram_percent) / len(self.ram_percent)
        max_pct = max(self.ram_percent)
        ax2.axhline(
            mean_pct, color="green", linestyle="--", label=f"Mean: {mean_pct:.1f}%"
        )
        ax2.axhline(max_pct, color="red", linestyle="--", label=f"Max: {max_pct:.1f}%")
        ax2.legend(loc="upper right")

        # Format bottom x-axis to show HH:MM
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

        # Add summary text
        total_elapsed = elapsed_seconds[-1] if elapsed_seconds else 0
        duration_str = format_elapsed_time(total_elapsed)
        summary = (
            f"Duration: {duration_str} | "
            f"Samples: {len(self.timestamps)} | "
            f"Interval: {self.interval}s"
        )
        fig.text(0.5, 0.02, summary, ha="center", fontsize=10, style="italic")

        plt.tight_layout(rect=[0, 0.03, 1, 1])

        # Save plot (overwriting previous)
        plt.savefig(self.plot_file, dpi=150, bbox_inches="tight")

        if show_message:
            print(f"✅ Plot saved to: {self.plot_file}")

            # Show statistics
            print(f"\n📊 Statistics:")
            print(f"   Duration: {duration_str}")
            print(f"   Samples: {len(self.timestamps)}")
            print(
                f"   RAM Used (GB) - Min: {min_gb:.2f}, Mean: {mean_gb:.2f}, Max: {max_gb:.2f}"
            )
            print(
                f"   RAM Used (%)  - Min: {min(self.ram_percent):.1f}, "
                f"Mean: {mean_pct:.1f}, Max: {max_pct:.1f}"
            )

        plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Monitor RAM usage over time",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Examples:
            # Monitor for 1 hour with 5-second intervals, update plot every minute
            python monitor_ram.py --interval 5 --duration 60 --plot-every 60
            
            # Monitor endlessly with 1-second intervals, update plot every 30 seconds
            python monitor_ram.py --interval 1 --endless --plot-every 30
            
            # Custom output file, disable live plotting
            python monitor_ram.py --interval 10 --duration 10 --output my_ram_log.csv --plot-every 0
                    """,
    )

    parser.add_argument(
        "--interval",
        type=float,
        default=5,
        help="Measurement interval in seconds (default: 5)",
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Total monitoring duration in minutes (default: None/endless)",
    )

    parser.add_argument(
        "--endless",
        action="store_true",
        help="Monitor endlessly until Ctrl+C (same as --duration None)",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file path (default: ram_usage_<timestamp>.csv)",
    )

    parser.add_argument(
        "--plot-every",
        type=float,
        default=60,
        help="Update plot every N seconds (default: 60, 0 to disable live plotting)",
    )

    args = parser.parse_args()

    # Handle --endless flag
    if args.endless:
        args.duration = None

    # Handle --plot-every 0 (disable live plotting)
    if args.plot_every == 0:
        args.plot_every = None

    # Create and run monitor
    monitor = RAMMonitor(
        interval=args.interval,
        duration=args.duration,
        output_file=args.output,
        plot_every=args.plot_every,
    )

    try:
        monitor.monitor()
    except KeyboardInterrupt:
        print("\n\n🛑 Interrupted by user.")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback

        traceback.print_exc()
    finally:
        print(f"\n✅ Data saved to: {monitor.output_file}")
        print(f"✅ Plot saved to: {monitor.plot_file}")
        print("\n👋 Monitoring complete!")


if __name__ == "__main__":
    main()
