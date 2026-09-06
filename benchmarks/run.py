import argparse
import json
import os
import subprocess
import sys

from .cases import all_cases, get_case
from .code import all_implementations, get_implementation
from .utils import run_benchmark


try:
    import torch

    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Benchmark neighbor list implementations"
    )
    available = ", ".join(all_implementations())
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for benchmark results (default: no output)",
    )
    parser.add_argument(
        "--case",
        default="all",
        help="Benchmark case(s) to run (default: all available cases)",
    )
    parser.add_argument(
        "--device",
        nargs="*",
        default=["all"],
        help="List of devices to run on (default: all supported devices)",
    )
    parser.add_argument(
        "implementations",
        nargs="*",
        default=["all"],
        help=(
            f"List of implementations to benchmark (default: all). The available "
            f"implementations are: {available}"
        ),
    )
    parser.add_argument(
        "--single",
        nargs=6,
        metavar=("IMPL", "DEVICE", "CASE", "IDX", "CUTOFF", "FULL_LIST"),
        help="Run a single benchmark in a subprocess for isolation",
    )
    return parser.parse_args(argv)


def select_tasks(arguments, devices):
    """Expand the requested implementations into (implementation, device) tasks."""
    algorithms = list(arguments)
    if algorithms == ["all"]:
        algorithms = all_implementations()

    explicit = not (not devices or devices == ["all"])
    if not devices or devices == ["all"]:
        devices = ["cpu", "cuda"]
    for device in devices:
        if device not in ("cpu", "cuda"):
            raise ValueError(f"Unknown device: {device}")
        if device == "cuda" and not CUDA_AVAILABLE:
            if explicit:
                raise ValueError("cuda is not available on this machine")
            continue

    tasks = []
    for algorithm in algorithms:
        implementation = get_implementation(algorithm)
        for device in implementation.devices:
            if device not in devices:
                continue
            if device == "cuda" and not CUDA_AVAILABLE:
                continue
            tasks.append((algorithm, device))
    return tasks


def check_pairs(impl, device, n_pairs, reference, case_name, idx, cutoff, full_list):
    """Compare an implementation pair count against the vesin reference."""
    if n_pairs != reference:
        print(
            f"WARNING: {impl} ({device}) reported {n_pairs} pairs, "
            f"expected {reference} (vesin) - case '{case_name}' #{idx}, "
            f"cutoff {cutoff}, full_list={full_list}",
            file=sys.stderr,
        )


def main(argv=None):
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args = parse_args(argv)

    if args.single is not None:
        impl, device, case_name, idx, cutoff, full_list = args.single
        structure = get_case(case_name).structures()[int(idx)]
        timing, n_pairs = run_benchmark(
            impl, device, structure.build(), float(cutoff), full_list == "True"
        )
        if n_pairs is None:
            print("nan nan")
        else:
            print(f"{n_pairs} {timing}")
        return

    if args.case == "all":
        cases = all_cases()
    else:
        cases = [get_case(args.case)]

    tasks = select_tasks(args.implementations, args.device)
    name_width = max(len(f"{impl} ({device})") for impl, device in tasks)

    data = {}
    for case in cases:
        case_data = {"n_atoms": []}
        for impl, device in tasks:
            if impl not in case_data:
                case_data[impl] = {"version": get_implementation(impl).version()}
            case_data[impl][device] = {}
            for option in case.options:
                list_kind = "full" if option.full_list else "half"
                case_data[impl][device][f"cutoff_{option.cutoff}_{list_kind}"] = []
        data[case.name] = case_data

    vesin_pairs = {}
    pending = {}
    for case in cases:
        too_slow = set()
        for idx, structure in enumerate(case.structures()):
            atoms = structure.build()
            print(f"\n# Benchmarking case '{case.name}' with {len(atoms)} atoms")
            data[case.name]["n_atoms"].append(len(atoms))

            for option in case.options:
                list_kind = "full" if option.full_list else "half"
                print(f"Cutoff: {option.cutoff} Å, {list_kind} list")

                for impl, device in tasks:
                    if option.full_list not in get_implementation(impl).full_list:
                        continue

                    key = (case.name, idx, option.cutoff, option.full_list)
                    skip_key = (impl, device, option.cutoff, option.full_list)
                    if skip_key in too_slow:
                        data[case.name][impl][device][
                            f"cutoff_{option.cutoff}_{list_kind}"
                        ].append(float("nan"))
                        print(
                            f"    {f'{impl} ({device})':{name_width}} skipped, too slow"
                        )
                        continue

                    result = subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "benchmarks.run",
                            "--single",
                            impl,
                            device,
                            case.name,
                            str(idx),
                            str(option.cutoff),
                            str(option.full_list),
                        ],
                        cwd=root,
                        capture_output=True,
                        text=True,
                    )

                    if result.stderr:
                        for line in result.stderr.strip().splitlines():
                            print(f"+++ {line}", file=sys.stderr)

                    timing = float("nan")
                    n_pairs = None
                    if result.returncode != 0 or not result.stdout.strip():
                        pass
                    else:
                        tokens = result.stdout.strip().split()
                        if len(tokens) == 2:
                            try:
                                n_pairs = int(tokens[0])
                                timing = float(tokens[1])
                            except ValueError:
                                n_pairs = None
                                timing = float("nan")
                    data[case.name][impl][device][
                        f"cutoff_{option.cutoff}_{list_kind}"
                    ].append(timing)
                    label = f"{impl} ({device})"
                    if n_pairs is None:
                        print(f"    {label:{name_width}} did not run")
                    else:
                        print(f"    {label:{name_width}} took {1e3 * timing:.4f} ms")

                    if timing > 10.0:
                        too_slow.add(skip_key)
                        print(f"    {label:{name_width}} too slow, skipping for next")

                    if n_pairs is None:
                        continue

                    if impl == "vesin":
                        if key not in vesin_pairs:
                            vesin_pairs[key] = n_pairs
                            for p_impl, p_device, p_npairs in pending.pop(key, []):
                                check_pairs(p_impl, p_device, p_npairs, n_pairs, *key)
                    else:
                        reference = vesin_pairs.get(key)
                        if reference is None:
                            pending.setdefault(key, []).append((impl, device, n_pairs))
                        else:
                            check_pairs(impl, device, n_pairs, reference, *key)

    if args.output is not None:
        json.dump(data, open(args.output, "w"), indent=4)


if __name__ == "__main__":
    main()
