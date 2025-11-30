# main.py
# -------------------------------------------------------
# Top-level entry point for the BPS Question Dedup System
# -------------------------------------------------------

import argparse
from src.dedup_engine import main as dedup_run
from src.clustering import main as cluster_run

def run_all():
    print("\n==============================")
    print("  BPS QUESTION DEDUP SYSTEM")
    print("==============================\n")

    print("[1] Running semantic dedup engine...")
    dedup_run()

    print("\n[2] Running clustering...")
    cluster_run()

    print("\n--- ALL COMPLETED SUCCESSFULLY ---")
    print("Check folder: results/")
    print(" - similarity_pairs.csv")
    print(" - heatmap.png")
    print(" - clusters.json\n")


def main():
    parser = argparse.ArgumentParser(description="BPS Question Dedup System")
    parser.add_argument(
        "--mode",
        type=str,
        default="all",
        choices=["all", "dedup", "cluster"],
        help="Run full pipeline or selected module",
    )

    args = parser.parse_args()

    if args.mode == "all":
        run_all()
    elif args.mode == "dedup":
        print("[Run] Dedup only...")
        dedup_run()
    elif args.mode == "cluster":
        print("[Run] Cluster only...")
        cluster_run()


if __name__ == "__main__":
    main()
