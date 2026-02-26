"""
Run the complete application pipeline end-to-end.

Usage:
    python scripts/run_e2e.py              # Data -> Features -> Train
    python scripts/run_e2e.py --serve      # Same + start API and frontend
    python scripts/run_e2e.py --skip-train # Data + Features only (no training)
    python scripts/run_e2e.py --train-only # Train only (assume data/features exist)

Steps:
  1. Data: load season CSVs, clean, save to data/raw/matches.csv
  2. Features: build features, save to data/features/model_ready.csv
  3. Train: train models, save best to models/
  4. (Optional) Serve: start API on 8000 and frontend on 3000
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# Ensure project root on path
project_root = str(Path(__file__).resolve().parent.parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def run_data_pipeline():
    """Load and clean raw data."""
    from src.data.loader import run_pipeline
    from src.utils.config import load_config
    print("Step 1/3: Data pipeline (load + clean)...")
    run_pipeline(load_config())
    print("  Done -> data/raw/matches.csv\n")


def run_feature_pipeline():
    """Build features from matches."""
    from src.features.builder import run_pipeline
    from src.utils.config import load_config
    print("Step 2/3: Feature pipeline...")
    run_pipeline(load_config())
    print("  Done -> data/features/model_ready.csv\n")


def run_train():
    """Train models and save best."""
    print("Step 3/3: Training...")
    result = subprocess.run(
        [sys.executable, "scripts/train.py", "--no-mlflow"],
        cwd=project_root,
    )
    if result.returncode != 0:
        sys.exit(result.returncode)
    print("  Done -> models/\n")


def start_servers():
    """Start API and frontend in background; block until Ctrl+C."""
    import signal
    procs = []

    def cleanup(signum=None, frame=None):
        for p in procs:
            if p.poll() is None:
                p.terminate()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, cleanup)

    print("Starting API (port 8000)...")
    api = subprocess.Popen(
        [sys.executable, "scripts/run_api.py"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(api)
    time.sleep(1)
    if api.poll() is not None:
        print("  API failed to start. Check dependencies and port 8000.")
        sys.exit(1)

    print("Starting frontend (port 3000)...")
    front = subprocess.Popen(
        [sys.executable, "scripts/run_frontend.py"],
        cwd=project_root,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    procs.append(front)
    time.sleep(0.5)
    if front.poll() is not None:
        api.terminate()
        print("  Frontend failed to start. Check port 3000.")
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Application is running.")
    print("  Frontend: http://localhost:3000")
    print("  API:      http://localhost:8000")
    print("  API docs: http://localhost:8000/docs")
    print("=" * 60)
    print("Press Ctrl+C to stop both servers.\n")

    try:
        while True:
            if api.poll() is not None or front.poll() is not None:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    cleanup()


def main():
    parser = argparse.ArgumentParser(description="Run EPL Predictor pipeline and optionally serve")
    parser.add_argument("--serve", action="store_true", help="After pipeline, start API and frontend")
    parser.add_argument("--skip-train", action="store_true", help="Only run data + features (no training)")
    parser.add_argument("--train-only", action="store_true", help="Only run training (assume data/features exist)")
    args = parser.parse_args()

    print("=" * 60)
    print("EPL PREDICTOR - END-TO-END")
    print("=" * 60 + "\n")

    if args.train_only:
        run_train()
    else:
        run_data_pipeline()
        run_feature_pipeline()
        if not args.skip_train:
            run_train()

    if args.serve:
        start_servers()
    else:
        print("Pipeline complete. To start the app:")
        print("  Terminal 1: python scripts/run_api.py")
        print("  Terminal 2: python scripts/run_frontend.py")
        print("  Then open: http://localhost:3000")


if __name__ == "__main__":
    main()
