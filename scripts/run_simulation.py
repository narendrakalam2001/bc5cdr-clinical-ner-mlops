# ============================================================
# RUN SIMULATION — BC5CDR Clinical NER System
# ============================================================
import sys, os, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from simulation.clinical_text_simulator import simulate_clinical_texts, simulate_all_scenarios


def main():
    parser = argparse.ArgumentParser(description="BC5CDR NER Simulator")
    parser.add_argument("--n",        type=int,  default=30)
    parser.add_argument("--scenario", type=str,  default="random",
                        choices=["random","normal","ade_report","multi_chemical","all"])
    args = parser.parse_args()

    if args.scenario == "all":
        simulate_all_scenarios(n_per_scenario=max(args.n // 3, 5))
    else:
        simulate_clinical_texts(n=args.n, scenario=args.scenario)


if __name__ == "__main__":
    main()
