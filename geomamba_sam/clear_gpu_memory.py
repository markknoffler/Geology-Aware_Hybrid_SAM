#!/usr/bin/env python3
"""
clear_gpu.py – free as much NVIDIA GPU memory as possible.

Steps:
  1. Release this Python process’ CUDA cache      (Torch only)
  2. Kill every remaining process using the GPU   (needs SIGKILL)
  3. Reset all GPUs via nvidia-smi                (needs sudo)
"""
import os, signal, subprocess, sys, gc

def free_own_torch_cache():
    """
    Empty PyTorch’s caching allocator and run Python’s GC.
    Works only if PyTorch is installed and the current process
    owns the allocations.  Leaves the CUDA context itself alive.
    """
    try:
        import torch
        torch.cuda.empty_cache()            # releases cached blocks[1]
    except ModuleNotFoundError:
        pass
    gc.collect()

def kill_external_gpu_processes():
    """
    Find every PID that holds a compute context on any NVIDIA GPU
    and shoot it with SIGKILL.  Skips our own pid.
    """
    try:
        # Ask nvidia-smi for the list of PIDs that own compute contexts
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            text=True
        )
        for pid_str in filter(None, out.splitlines()):
            pid = int(pid_str)
            if pid != os.getpid():
                try:
                    os.kill(pid, signal.SIGKILL)   # abrupt but effective[2]
                except ProcessLookupError:
                    pass  # process already vanished
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("nvidia-smi not found – skipping external-process kill step", file=sys.stderr)

def reset_all_gpus():
    """
    Ask the driver to reset every GPU.  Requires super-user privileges and
    will nuke *all* remaining contexts (graphics or compute)[4].
    """
    try:
        subprocess.run(["sudo", "nvidia-smi", "--gpu-reset"],
                       check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except FileNotFoundError:
        print("nvidia-smi not found – cannot issue GPU reset", file=sys.stderr)

if __name__ == "__main__":
    free_own_torch_cache()
    kill_external_gpu_processes()
    reset_all_gpus()
    print("Attempted to clear all GPU memory.  Verify with `nvidia-smi`.")
