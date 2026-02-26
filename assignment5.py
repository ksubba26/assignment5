from __future__ import annotations

import random
import time
from statistics import median
from typing import List, Callable, Dict, Tuple

#Partition (Lomuto)
def partition_lomuto(arr: List[int], low: int, high: int) -> int:
    pivot = arr[high]
    i = low - 1
    for j in range(low, high):
        if arr[j] <= pivot:
            i += 1
            arr[i], arr[j] = arr[j], arr[i]
    arr[i + 1], arr[high] = arr[high], arr[i + 1]
    return i + 1

# Deterministic Quicksort (Iterative)

def quicksort_deterministic(arr: List[int]) -> None:
    """
    Deterministic quicksort (last element pivot), implemented ITERATIVELY to avoid recursion depth errors.
    Worst-case time still O(n^2) on sorted/reverse inputs, but no RecursionError.
    """
    if len(arr) <= 1:
        return

    stack: List[Tuple[int, int]] = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()
        if low >= high:
            continue

        p = partition_lomuto(arr, low, high)

        #Push larger segment first so smaller is processed sooner (keeps stack smaller)
        left = (low, p - 1)
        right = (p + 1, high)

        left_size = left[1] - left[0]
        right_size = right[1] - right[0]

        if left_size > right_size:
            if left[0] < left[1]:
                stack.append(left)
            if right[0] < right[1]:
                stack.append(right)
        else:
            if right[0] < right[1]:
                stack.append(right)
            if left[0] < left[1]:
                stack.append(left)

# Randomized Quicksort (Iterative)

def quicksort_randomized(arr: List[int], seed: int | None = None) -> None:
    """
    Randomized quicksort (random pivot), implemented ITERATIVELY to avoid recursion depth errors.
    Expected time O(n log n) for any input distribution; worst-case still O(n^2) but unlikely.
    """
    if len(arr) <= 1:
        return

    rng = random.Random(seed)
    stack: List[Tuple[int, int]] = [(0, len(arr) - 1)]

    while stack:
        low, high = stack.pop()
        if low >= high:
            continue

        pivot_index = rng.randint(low, high)
        arr[pivot_index], arr[high] = arr[high], arr[pivot_index]

        p = partition_lomuto(arr, low, high)

        left = (low, p - 1)
        right = (p + 1, high)

        left_size = left[1] - left[0]
        right_size = right[1] - right[0]

        if left_size > right_size:
            if left[0] < left[1]:
                stack.append(left)
            if right[0] < right[1]:
                stack.append(right)
        else:
            if right[0] < right[1]:
                stack.append(right)
            if left[0] < left[1]:
                stack.append(left)

# Benchmarking

def make_input(dist: str, n: int, rng: random.Random) -> List[int]:
    if dist == "random":
        return [rng.randint(0, 10**9) for _ in range(n)]
    if dist == "sorted":
        return list(range(n))
    if dist == "reverse":
        return list(range(n, 0, -1))
    raise ValueError(f"Unknown distribution: {dist}")


def time_sort(fn: Callable[[List[int]], None], data: List[int]) -> float:
    arr = list(data)
    t0 = time.perf_counter()
    fn(arr)
    t1 = time.perf_counter()
    assert arr == sorted(data)
    return t1 - t0


def run_benchmarks(
    sizes: List[int],
    dists: List[str],
    trials: int = 7,
    seed: int = 123
) -> Dict[Tuple[int, str], Dict[str, float]]:
    rng = random.Random(seed)

    def det(a: List[int]) -> None:
        quicksort_deterministic(a)

    def rnd(a: List[int]) -> None:
        quicksort_randomized(a, seed=seed)

    results: Dict[Tuple[int, str], Dict[str, float]] = {}

    for n in sizes:
        for dist in dists:
            base = make_input(dist, n, rng)

            det_times = [time_sort(det, base) for _ in range(trials)]
            rnd_times = [time_sort(rnd, base) for _ in range(trials)]

            results[(n, dist)] = {
                "deterministic_median_s": median(det_times),
                "randomized_median_s": median(rnd_times),
            }

    return results


def print_results(results: Dict[Tuple[int, str], Dict[str, float]]) -> None:
    print(f"{'n':>8}  {'dist':>8}  {'det (s)':>12}  {'rand (s)':>12}  {'speedup':>10}")
    print("-" * 60)
    for (n, dist), r in sorted(results.items()):
        det = r["deterministic_median_s"]
        rnd = r["randomized_median_s"]
        speedup = (det / rnd) if rnd > 0 else float("inf")
        print(f"{n:>8}  {dist:>8}  {det:>12.6f}  {rnd:>12.6f}  {speedup:>10.2f}x")

# Main

if __name__ == "__main__":
    demo = [10, 7, 8, 9, 1, 5]
    a = demo[:]
    b = demo[:]

    quicksort_deterministic(a)
    quicksort_randomized(b, seed=123)

    print("Demo input:        ", demo)
    print("Deterministic QS:  ", a)
    print("Randomized QS:     ", b)
    print()

    sizes = [1000, 2000, 5000, 10000]
    dists = ["random", "sorted", "reverse"]

    results = run_benchmarks(sizes, dists, trials=7, seed=123)
    print_results(results)