import subprocess
import sys
import time
import shutil
from pathlib import Path
from tqdm import tqdm

# ---------------- CONFIG ----------------
REMOTE = "bbd0953@fs08"
REMOTE_DIR = "/hs/fs08/data/group-brueggen/tmartinez/diffusion/image_data/LOFAR/micromaps-arrow/micromaps-DR3-opt-1024px-spacing=1/train.arrow"
LOCAL_DIR = "/hs/babbage/data/group-brueggen/tmartinez/diffusion/image_data_local/LOFAR/micromaps-arrow/micromaps-DR3-opt-1024px-spacing=1"
BLOCK = 10 * 1024**2  # n * 1 MB
GB = 1024**3
compress = False
# ----------------------------------------


def get_remote_size():
    cmd = ["ssh", REMOTE, "du", "-sb", REMOTE_DIR]
    out = subprocess.check_output(cmd, text=True)
    return int(out.split()[0])


def _check_failed(proc: subprocess.Popen, name: str, err_stream):
    rc = proc.poll()
    if rc is not None and rc != 0:
        err = err_stream.read().decode(errors="replace") if err_stream else ""
        raise RuntimeError(f"{name} failed (rc={rc}). {err}")


def main():
    print("Calculating remote size…", file=sys.stderr)
    total_size = get_remote_size()
    print(f"Total size to transfer: {total_size / GB:.2f} GB", file=sys.stderr)
    ssh_cmd = [
        "ssh",
        REMOTE,
        f"tar -C {Path(REMOTE_DIR).parent} -cf - {Path(REMOTE_DIR).name}"
        + (" | gzip" if compress else ""),
    ]

    tar_cmd = ["tar", "xzf" if compress else "xf", "-", "-C", LOCAL_DIR]

    ssh = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    tar = subprocess.Popen(tar_cmd, stdin=subprocess.PIPE)

    try:
        with tqdm(
            total=total_size,
            unit="B",
            unit_scale=True,
            desc="Transferring",
            file=sys.stderr,
            dynamic_ncols=True,
            ncols=40,
            smoothing=0.01,
        ) as bar:
            while True:
                _check_failed(ssh, "ssh/tar(remote)", ssh.stderr)
                _check_failed(tar, "tar(local)", tar.stderr)

                chunk = ssh.stdout.read(BLOCK)
                if not chunk:
                    break

                try:
                    tar.stdin.write(chunk)
                except BrokenPipeError:
                    _check_failed(tar, "tar(local)", tar.stderr)
                    raise RuntimeError("Local tar stdin closed unexpectedly.")

                bar.update(len(chunk))
    finally:
        if tar.stdin and not tar.stdin.closed:
            tar.stdin.close()

    ssh.wait()
    tar.wait()
    _check_failed(ssh, "ssh/tar(remote)", ssh.stderr)
    _check_failed(tar, "tar(local)", tar.stderr)

    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
