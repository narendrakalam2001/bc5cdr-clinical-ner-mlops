# ============================================================
# CLINICAL TEXT SIMULATOR — BC5CDR Clinical NER System
# Scenarios: normal, ade_report, multi_chemical
# ============================================================

import requests, random, time, os

API_URL = os.getenv(
    "NER_API_URL", "http://localhost:8000"
) + "/predict"

# ── BC5CDR vocabulary (PubMed biomedical style) ───────────────
CHEMICALS = [
    "Cyclosporin", "Cisplatin", "Methotrexate", "Amoxicillin",
    "Doxorubicin", "Carboplatin", "Tacrolimus", "Azathioprine",
    "Cyclophosphamide", "Vincristine", "Bleomycin", "Chlorambucil",
    "Hydroxychloroquine", "Sulfasalazine", "Leflunomide", "Infliximab",
]

DISEASES = [
    "nephrotoxicity", "hepatotoxicity", "ototoxicity", "cardiotoxicity",
    "peripheral neuropathy", "myelosuppression", "thrombocytopenia",
    "leukopenia", "anemia", "hypertension", "diabetes mellitus",
    "rheumatoid arthritis", "systemic lupus erythematosus",
    "Crohn's disease", "ulcerative colitis", "pulmonary fibrosis",
]

CONTEXTS = [
    "in patients with", "in subjects with", "associated with",
    "was administered to patients with", "in the treatment of",
    "was used for", "has been reported in patients with",
]

OUTCOMES = [
    "resulting in significant clinical improvement",
    "with dose-dependent adverse effects",
    "leading to treatment discontinuation",
    "requiring dose reduction",
    "with manageable side effects",
    "showing complete response",
]


def generate_clinical_text(scenario: str = "normal") -> dict:
    """
    Generates BC5CDR-style biomedical text.

    Scenarios:
        normal       — standard drug-disease sentence (PubMed style)
        ade_report   — adverse drug event (Chemical causes Disease)
        multi_chemical — multiple chemicals in one sentence
    """
    if scenario == "normal":
        chem    = random.choice(CHEMICALS)
        disease = random.choice(DISEASES)
        context = random.choice(CONTEXTS)
        outcome = random.choice(OUTCOMES)

        templates = [
            f"{chem} {context} {disease} {outcome}.",
            f"The administration of {chem} in patients with {disease} showed {outcome.split()[1:3]}.",
            f"We investigated the efficacy of {chem} {context} {disease}.",
            f"{chem}-treated patients with {disease} demonstrated significant changes.",
        ]
        text = random.choice(templates)
        return {"scenario": "normal", "text": text,
                "expected": [chem, disease]}

    elif scenario == "ade_report":
        chem    = random.choice(CHEMICALS)
        disease = random.choice(DISEASES)
        sev     = random.choice(["severe", "moderate", "mild", "acute"])

        templates = [
            f"{chem}-induced {sev} {disease} was observed in {random.randint(3,15)} patients.",
            f"We report a case of {sev} {disease} following {chem} therapy.",
            f"{chem} administration resulted in {sev} {disease} in the study cohort.",
            f"Adverse reaction: {sev} {disease} attributed to {chem} treatment.",
        ]
        text = random.choice(templates)
        return {"scenario": "ade_report", "text": text,
                "expected": [chem, disease], "is_ade": True}

    else:  # multi_chemical
        chem1   = random.choice(CHEMICALS)
        chem2   = random.choice([c for c in CHEMICALS if c != chem1])
        disease = random.choice(DISEASES)

        templates = [
            f"Combination therapy with {chem1} and {chem2} in {disease} patients.",
            f"The synergistic effect of {chem1} plus {chem2} on {disease}.",
            f"Patients received {chem1} followed by {chem2} for treatment of {disease}.",
            f"We compared {chem1} versus {chem2} in patients with {disease}.",
        ]
        text = random.choice(templates)
        return {"scenario": "multi_chemical", "text": text,
                "expected": [chem1, chem2, disease]}


def send_text(text_info: dict, idx: int) -> None:
    try:
        response = requests.post(
            API_URL,
            json={"text": text_info["text"], "return_confidence": True},
            timeout=30
        )
        if response.status_code == 200:
            result   = response.json()
            entities = result.get("entities", [])
            review   = result.get("requires_review", False)
            ent_summary = ", ".join(
                f"{e['entity_type']}:{e['text'][:15]}" for e in entities[:4]
            )
            print(
                f"[{idx+1:03d}] SCENARIO={text_info['scenario']:<15} | "
                f"entities={len(entities):2d} | "
                f"review={'YES' if review else 'NO ':3s} | "
                f"{ent_summary}"
            )
            if result.get("alert"):
                print(f"       ⚠️  {result['alert']}")
        else:
            print(f"[{idx+1:03d}] API error: HTTP {response.status_code}")
    except Exception as e:
        print(f"[{idx+1:03d}] Connection error: {e}")


def simulate_clinical_texts(n: int = 30, scenario: str = "random") -> None:
    print(f"\nBC5CDR Simulator | n={n} | scenario={scenario}")
    print("-" * 70)

    scenario_mix = {
        "random":        ["normal"] * 5 + ["ade_report"] * 3 + ["multi_chemical"] * 2,
        "normal":        ["normal"],
        "ade_report":    ["ade_report"],
        "multi_chemical":["multi_chemical"],
    }
    choices = scenario_mix.get(scenario, ["normal"])

    for i in range(n):
        chosen   = random.choice(choices)
        text_info = generate_clinical_text(scenario=chosen)
        send_text(text_info, i)
        time.sleep(0.5)

    print("-" * 70)
    print(f"Simulation complete: {n} texts sent")


def simulate_all_scenarios(n_per_scenario: int = 10) -> None:
    for sc in ["normal", "ade_report", "multi_chemical"]:
        print(f"\n{'='*70}")
        print(f"SCENARIO: {sc.upper()}")
        print("=" * 70)
        simulate_clinical_texts(n=n_per_scenario, scenario=sc)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="BC5CDR Clinical NER Simulator")
    parser.add_argument("--n",        type=int, default=30)
    parser.add_argument("--scenario", type=str, default="random",
                        choices=["random","normal","ade_report","multi_chemical","all"])
    args = parser.parse_args()

    if args.scenario == "all":
        simulate_all_scenarios(n_per_scenario=max(args.n // 3, 5))
    else:
        simulate_clinical_texts(n=args.n, scenario=args.scenario)
