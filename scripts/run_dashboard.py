# ============================================================
# RUN DASHBOARD — BC5CDR Clinical NER System
# ============================================================
import sys, os, argparse, subprocess
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def main():
    parser = argparse.ArgumentParser(description="BC5CDR NER Monitoring Dashboard")
    parser.add_argument("--port", type=int, default=8501)
    args = parser.parse_args()

    dashboard_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "monitoring", "monitoring_dashboard.py")
    )
    print(f"\n🧬 BC5CDR NER Dashboard starting on http://localhost:{args.port}\n")
    subprocess.run([sys.executable, "-m", "streamlit", "run", dashboard_path,
                    "--server.port", str(args.port), "--server.headless", "true"])


if __name__ == "__main__":
    main()
