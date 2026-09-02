#!/usr/bin/env python3
"""
GPU memory holder with visible, periodic GPU-utilization fluctuations.

Example:
    CUDA_VISIBLE_DEVICES=2 python gpu_holder_v2.py

Default behavior:
    - hold ~30 GiB VRAM
    - continuously create short BF16 GEMM workloads
    - alternate among several compute duty cycles so nvidia-smi shows
      clearly changing GPU utilization instead of sitting near 0%

Stop with Ctrl+C.
"""

import argparse
import signal
import time

import torch


def gib(x: int) -> float:
    return x / (1024 ** 3)


def allocate_gpu_memory(target_gib: float, chunk_gib: float = 1.0):
    """Allocate and touch CUDA memory so it remains resident."""
    target_bytes = int(target_gib * 1024 ** 3)
    chunk_bytes = int(chunk_gib * 1024 ** 3)

    holders = []
    allocated = 0

    print(f"[MEM] target = {target_gib:.2f} GiB")

    while allocated < target_bytes:
        nbytes = min(chunk_bytes, target_bytes - allocated)

        # uint8 => 1 byte per element.
        x = torch.empty(
            nbytes,
            dtype=torch.uint8,
            device="cuda",
        )

        # Touch memory so the allocation is committed.
        x.fill_(1)

        holders.append(x)
        allocated += nbytes

        print(
            f"\r[MEM] allocated={gib(torch.cuda.memory_allocated()):.2f} GiB "
            f"reserved={gib(torch.cuda.memory_reserved()):.2f} GiB",
            end="",
            flush=True,
        )

    torch.cuda.synchronize()
    print()

    return holders


class ComputeLoad:
    """
    Reusable BF16 GEMM workload.

    Matrices are allocated once and repeatedly multiplied.  Synchronizing
    after each GEMM makes the requested busy/idle duty cycle visible to
    nvidia-smi instead of merely queuing asynchronous CUDA work.
    """

    def __init__(self, matrix_size: int):
        self.matrix_size = matrix_size

        print(
            f"[LOAD] allocating BF16 GEMM matrices "
            f"{matrix_size} x {matrix_size}"
        )

        self.a = torch.randn(
            matrix_size,
            matrix_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        self.b = torch.randn(
            matrix_size,
            matrix_size,
            device="cuda",
            dtype=torch.bfloat16,
        )

        # Warm-up kernels / cuBLAS.
        for _ in range(3):
            c = self.a @ self.b
            self.a, self.b = self.b, c

        torch.cuda.synchronize()

    def work_for(self, seconds: float) -> int:
        """Run synchronized GEMMs for approximately `seconds`."""
        start = time.perf_counter()
        iterations = 0

        while time.perf_counter() - start < seconds:
            c = self.a @ self.b

            # Synchronize each iteration so GPU work happens during this
            # busy interval rather than being queued and spilling into idle.
            torch.cuda.synchronize()

            self.a, self.b = self.b, c
            iterations += 1

        return iterations


def run_duty_window(
    workload: ComputeLoad,
    window_seconds: float,
    duty: float,
    slice_seconds: float,
    should_stop,
):
    """
    Approximate a utilization level by alternating short compute and sleep
    slices within a window.

    Example:
        duty=0.70, slice=1.0
        -> ~0.70 s compute + ~0.30 s idle per second.

    This does not guarantee an exact nvidia-smi percentage; it deliberately
    produces sustained, visible utilization instead of a near-zero graph.
    """

    duty = max(0.0, min(1.0, duty))

    busy_time = slice_seconds * duty
    idle_time = slice_seconds - busy_time

    start = time.perf_counter()
    iterations = 0

    while time.perf_counter() - start < window_seconds and not should_stop():
        remaining = window_seconds - (
            time.perf_counter() - start
        )

        if remaining <= 0:
            break

        current_slice = min(slice_seconds, remaining)

        current_busy = current_slice * duty
        current_idle = current_slice - current_busy

        if current_busy > 0:
            iterations += workload.work_for(current_busy)

        if current_idle > 0 and not should_stop():
            time.sleep(current_idle)

    return iterations


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--target-gb",
        type=float,
        default=30.0,
        help="VRAM to keep allocated in GiB.",
    )

    parser.add_argument(
        "--matrix-size",
        type=int,
        default=8192,
        help="BF16 GEMM matrix dimension.",
    )

    parser.add_argument(
        "--window",
        type=float,
        default=8.0,
        help="Seconds spent at each utilization level.",
    )

    parser.add_argument(
        "--slice",
        type=float,
        default=1.0,
        help="Duty-cycle control interval in seconds.",
    )

    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    total_gb = gib(props.total_memory)

    if args.target_gb > total_gb - 3.0:
        raise ValueError(
            f"Requested {args.target_gb:.1f} GiB on a "
            f"{total_gb:.1f} GiB GPU. Leave at least ~3 GiB free."
        )

    stop = False

    def handle_stop(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_stop)
    signal.signal(signal.SIGTERM, handle_stop)

    print("=" * 76)
    print("GPU MEMORY HOLDER + VISIBLE UTILIZATION FLUCTUATION")
    print("=" * 76)
    print("GPU:", torch.cuda.get_device_name(device))
    print(f"VRAM: {total_gb:.2f} GiB")
    print(f"Hold target: {args.target_gb:.2f} GiB")
    print(f"GEMM matrix: {args.matrix_size} x {args.matrix_size} BF16")
    print(f"Each load level lasts: {args.window:.1f} s")
    print("Ctrl+C to stop.")
    print()

    torch.cuda.reset_peak_memory_stats()

    holders = None
    workload = None

    # Repeating load pattern. These are duty cycles, not guaranteed exact
    # nvidia-smi utilization percentages.
    pattern = [
        ("LOW", 0.20),
        ("MEDIUM", 0.50),
        ("HIGH", 0.85),
        ("MEDIUM", 0.45),
    ]

    try:
        holders = allocate_gpu_memory(
            target_gib=args.target_gb,
        )

        workload = ComputeLoad(
            matrix_size=args.matrix_size,
        )

        print(
            f"[READY] allocated="
            f"{gib(torch.cuda.memory_allocated()):.2f} GiB, "
            f"reserved="
            f"{gib(torch.cuda.memory_reserved()):.2f} GiB"
        )
        print()
        print(
            "Open another terminal with: "
            "watch -n 0.5 nvidia-smi"
        )
        print()

        cycle = 0

        while not stop:
            cycle += 1
            print(f"========== cycle {cycle} ==========")

            for label, duty in pattern:
                if stop:
                    break

                print(
                    f"[{label:6s}] target duty ~{int(duty * 100):2d}% "
                    f"for {args.window:.1f}s"
                )

                iters = run_duty_window(
                    workload=workload,
                    window_seconds=args.window,
                    duty=duty,
                    slice_seconds=args.slice,
                    should_stop=lambda: stop,
                )

                print(
                    f"         GEMM iterations={iters}, "
                    f"allocated="
                    f"{gib(torch.cuda.memory_allocated()):.2f} GiB"
                )

    finally:
        print()
        print("[EXIT] releasing GPU resources...")

        workload = None

        if holders is not None:
            holders.clear()

        torch.cuda.empty_cache()
        torch.cuda.synchronize()

        print(
            f"[EXIT] allocated="
            f"{gib(torch.cuda.memory_allocated()):.2f} GiB, "
            f"reserved="
            f"{gib(torch.cuda.memory_reserved()):.2f} GiB"
        )


if __name__ == "__main__":
    main()
