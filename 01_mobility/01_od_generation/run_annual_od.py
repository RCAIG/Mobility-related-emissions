"""Generate missing 2018-2023 OD matrices with year-matched inputs.

Run from the GlODGen project root after the companion modules are deployed.
Disjoint city shards permit one process per GPU without shared log files.
"""

import argparse
import csv
import os
import signal
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


WAYBACK_IDS = {2018: 10768, 2019: 18063, 2020: 32645,
               2021: 9812, 2022: 26083, 2023: 47963}
GROUPS = ("high_income", "low_income", "lower_middle", "upper_middle")
RUN_GROUPS = GROUPS + ("unassigned", "all")
PROJECT = Path(__file__).resolve().parent
SHAPES = PROJECT / "assets/example_data/my_test/shp"
OUTPUT = Path("${SERVER_HOME}/result/od/annual_2018_2023")
FIELDS = ("timestamp_utc", "year", "group", "city_id", "status", "seconds", "error")


class CityTimeout(TimeoutError):
    pass


def timeout_handler(_signum, _frame):
    raise CityTimeout("city exceeded configured wall-time limit")


def completed(npy_path, csv_path, expected_size):
    if not npy_path.is_file() or not csv_path.is_file() or csv_path.stat().st_size == 0:
        return False
    try:
        matrix = np.load(npy_path, mmap_mode="r", allow_pickle=False)
        return matrix.shape == (expected_size, expected_size) and np.isfinite(matrix).all() and (matrix >= 0).all()
    except (OSError, ValueError):
        return False


def record(log_path, year, group, city, status, elapsed, error=""):
    new = not log_path.exists()
    with log_path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        if new:
            writer.writeheader()
        writer.writerow({"timestamp_utc": datetime.now(timezone.utc).isoformat(),
                         "year": year, "group": group, "city_id": city,
                         "status": status, "seconds": round(elapsed, 2),
                         "error": error[:1000]})


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, choices=tuple(WAYBACK_IDS), required=True)
    parser.add_argument("--group", choices=RUN_GROUPS, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1, 2, 3), required=True)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--city-timeout", type=int, default=1800,
                        help="Seconds per city; zero disables alarm (default 1800)")
    parser.add_argument("--sample-times", type=int, default=50)
    parser.add_argument("--max-cities", type=int, default=None,
                        help="Limit attempted cities for a controlled pilot")
    parser.add_argument("--charts", action="store_true", help="Render arc charts after OD is saved")
    parser.add_argument("--dry-run", action="store_true", help="Inspect paths/counts only")
    args = parser.parse_args()
    if args.city_timeout < 0 or args.sample_times < 1 or (args.max_cities is not None and args.max_cities < 1):
        parser.error("Timeout must be nonnegative, and sample/max-cities must be positive")
    if args.shard_count < 1 or not 0 <= args.shard_index < args.shard_count:
        parser.error("Shard index must be between zero and shard-count minus one")
    return args


def main():
    args = arguments()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    shape_dir = SHAPES / ("ALL" if args.group in ("unassigned", "all") else args.group)
    population = (PROJECT / "assets/example_data/my_test/tif" /
                  f"landscan-global-{args.year}-assets/landscan-global-{args.year}.tif")
    shapes = sorted(shape_dir.glob("*.shp"))
    if args.group == "unassigned":
        grouped = {shp.stem for group in GROUPS for shp in (SHAPES / group).glob("*.shp")}
        shapes = [shp for shp in shapes if shp.stem not in grouped]
    shapes = shapes[args.shard_index::args.shard_count]
    destination = args.output / f"{args.group}_{args.year}"
    print(f"year={args.year} Wayback={WAYBACK_IDS[args.year]} LandScan={population} "
          f"shapes={len(shapes)} output={destination} GPU={args.gpu} "
          f"shard={args.shard_index}/{args.shard_count}", flush=True)
    if not population.is_file() or not shapes:
        raise FileNotFoundError("Missing annual LandScan raster or group shapefiles")
    if args.dry_run:
        print("dry run: no output created, no model loaded", flush=True)
        return

    import geopandas as gpd
    import matplotlib.pyplot as plt
    from generate_od_annual.generator import Generator

    destination.mkdir(parents=True, exist_ok=True)
    log_path = destination / (f"run_log_shard{args.shard_index}.csv" if args.shard_count > 1
                              else "run_log.csv")
    generator = Generator()
    generator.set_satetoken(os.getenv("ESRI_TOKEN", ""))
    signal.signal(signal.SIGALRM, timeout_handler)
    succeeded = skipped = failed = 0
    attempted = 0

    for shp in shapes:
        city = shp.stem
        npy_path = destination / f"{city}_od_matrix.npy"
        csv_path = destination / f"{city}_od_matrix.csv"
        chart_path = destination / f"{city}_od_arc_chart.png"
        temp_npy = destination / f".{city}.{os.getpid()}.npy.tmp"
        temp_csv = destination / f".{city}.{os.getpid()}.csv.tmp"
        start = time.monotonic()
        try:
            if args.city_timeout:
                signal.alarm(args.city_timeout)
            # Only use an existing matrix when its dimensions match the current SHP.
            area = gpd.read_file(shp)
            if completed(npy_path, csv_path, len(area)):
                skipped += 1
                continue
            if args.max_cities is not None and attempted >= args.max_cities:
                break
            attempted += 1
            print(f"START {args.year} {args.group} {city}", flush=True)
            generator.set_annual_inputs(args.year, WAYBACK_IDS[args.year], city_id=city)
            generator.load_area(area)
            matrix = generator.generate(sample_times=args.sample_times)
            if matrix.shape != (len(area), len(area)) or not np.isfinite(matrix).all() or (matrix < 0).any():
                raise ValueError("Invalid OD shape or values")
            with temp_npy.open("wb") as stream:
                np.save(stream, matrix)
            np.savetxt(temp_csv, matrix, delimiter=",", fmt="%d")
            temp_npy.replace(npy_path)
            temp_csv.replace(csv_path)
            signal.alarm(0)
            succeeded += 1
            record(log_path, args.year, args.group, city, "success", time.monotonic() - start)
            print(f"DONE {city} total_OD={matrix.sum():.0f} elapsed={time.monotonic()-start:.1f}s", flush=True)
            if args.charts:
                try:
                    fig = generator.plot_arc_chart()
                    fig.savefig(chart_path, dpi=300, bbox_inches="tight")
                    plt.close(fig)
                except Exception as exc:
                    record(log_path, args.year, args.group, city, "chart_failed", 0, repr(exc))
        except Exception as exc:
            failed += 1
            error = f"{type(exc).__name__}: {exc}"
            record(log_path, args.year, args.group, city, "failed", time.monotonic() - start, error)
            print(f"FAILED {city}: {error}\n{traceback.format_exc(limit=2)}", flush=True)
        finally:
            signal.alarm(0)
            temp_npy.unlink(missing_ok=True)
            temp_csv.unlink(missing_ok=True)
    print(f"FINISHED year={args.year} group={args.group} "
          f"success={succeeded} existing={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
