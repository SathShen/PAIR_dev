#!/usr/bin/env python3
import argparse
import signal
import time

import torch


def gib(x):
    return x / (1024 ** 3)


def allocate_gpu_memory(target_gb: float, chunk_gb: float = 1.0):
    """
    Allocate and touch CUDA memory in chunks so it is actually committed.
    Uses uint8 so requested bytes ~= tensor bytes.
    """
    tensors = []
    target_bytes = int(target_gb * 1024 ** 3)
    chunk_bytes = int(chunk_gb * 1024 ** 3)

    allocated = 0
    print(f"[MEM] Target: {target_gb:.2f} GiB")

    while allocated < target_bytes:
        this_bytes = min(chunk_bytes, target_bytes - allocated)

        # uint8 = 1 byte / element
        x = torch.empty(this_bytes, dtype=torch.uint8, device="cuda")
        x.fill_(1)  # touch the allocation
        tensors.append(x)

        allocated += this_bytes
        print(
            f"\r[MEM] allocated={gib(torch.cuda.memory_allocated()):.2f} GiB, "
            f"reserved={gib(torch.cuda.memory_reserved()):.2f} GiB",
            end="",
            flush=True,
        )

    torch.cuda.synchronize()
    print()
    return tensors


def gpu_burst(seconds: float, matrix_size: int = 8192):
    """
    Create a short GEMM workload to make GPU utilization rise.
    BF16 keeps temporary memory modest while providing a real compute burst.
    """
    a = torch.randn(
        matrix_size, matrix_size,
        device="cuda",
        dtype=torch.bfloat16,
    )
    b = torch.randn(
        matrix_size, matrix_size,
        device="cuda",
        dtype=torch.bfloat16,
    )

    torch.cuda.synchronize()
    start = time.time()
    iters = 0

    while time.time() - start < seconds:
        c = a @ b
        # Change the inputs slightly so the loop is not completely identical.
        a, b = b, c
        iters += 1

    torch.cuda.synchronize()
    del a, b, c

    return iters


def main():
    parser = argparse.ArgumentParser(
        description="Reserve GPU memory and periodically create utilization bursts."
    )
    parser.add_argument("--target-gb", type=float, default=30.0)
    parser.add_argument("--idle", type=float, default=20.0,
                        help="Idle seconds between utilization bursts.")
    parser.add_argument("--busy", type=float, default=5.0,
                        help="Duration of each utilization burst.")
    parser.add_argument("--matrix-size", type=int, default=8192,
                        help="Square BF16 GEMM matrix size.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    props = torch.cuda.get_device_properties(0)
    total_gb = gib(props.total_memory)

    print("=" * 72)
    print("GPU MEMORY HOLDER + PERIODIC UTILIZATION BURST")
    print("=" * 72)
    print("GPU:", torch.cuda.get_device_name(0))
    print(f"Total VRAM: {total_gb:.2f} GiB")
    print(f"Requested hold: {args.target_gb:.2f} GiB")
    print(f"Pattern: idle {args.idle:.1f}s -> busy {args.busy:.1f}s -> repeat")
    print("Press Ctrl+C to exit and release memory.")
    print()

    # Leave some basic safety room for CUDA context + burst matrices.
    if args.target_gb > total_gb - 4:
        raise ValueError(
            f"--target-gb is too high for this GPU. "
            f"Use <= {total_gb - 4:.1f} GiB."
        )

    stop = False

    def handle_signal(signum, frame):
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    holders = allocate_gpu_memory(args.target_gb)

    print("[READY] GPU memory is being held.")
    print(
        f"[READY] allocated={gib(torch.cuda.memory_allocated()):.2f} GiB, "
        f"reserved={gib(torch.cuda.memory_reserved()):.2f} GiB"
    )
    print()

    cycle = 0

    try:
        while not stop:
            cycle += 1

            print(f"[Cycle {cycle}] idle for {args.idle:.1f}s ...")
            end_idle = time.time() + args.idle
            while time.time() < end_idle and not stop:
                time.sleep(min(1.0, end_idle - time.time()))

            if stop:
                break

            print(f"[Cycle {cycle}] GPU burst for {args.busy:.1f}s ...")
            iters = gpu_burst(
                seconds=args.busy,
                matrix_size=args.matrix_size,
            )
            print(
                f"[Cycle {cycle}] burst done, GEMM iterations={iters}, "
                f"allocated={gib(torch.cuda.memory_allocated()):.2f} GiB"
            )

    finally:
        print("\n[EXIT] Releasing GPU memory...")
        holders.clear()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(
            f"[EXIT] allocated={gib(torch.cuda.memory_allocated()):.2f} GiB, "
            f"reserved={gib(torch.cuda.memory_reserved()):.2f} GiB"
        )


if __name__ == "__main__":
    main()
