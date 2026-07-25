import json
import time
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent


def load_test_cases() -> list:
    cases_path = PROJECT_ROOT / "eval" / "test_cases.json"
    with open(cases_path, "r") as f:
        data = json.load(f)
    return data["cases"]


def score_response(response_text: str, case: dict) -> dict:
    """
    Score a single response against a test case.

    Three dimensions:
    1. Concept coverage: did the response mention expected concepts?
    2. Safety compliance: did it include required safety flags?
    3. Negative check: did it avoid things it should not say?

    Design decision: scoring is keyword-based, not LLM-based.
    This is intentional. LLM-based evaluation is more nuanced but
    adds cost, latency, and a second model's biases. Keyword scoring
    is transparent, reproducible, and auditable — important for a
    clinical tool where evaluation methodology will be scrutinized.
    The limitation (missing paraphrased concepts) is documented.
    """
    response_lower = response_text.lower()
    scores = {}

    # Score 1: Concept coverage
    expected_concepts = (
        case.get("expected_concepts", []) +
        case.get("expected_drugs", [])
    )

    if case.get("expected_regimen"):
        expected_concepts.append(case["expected_regimen"].lower())

    if expected_concepts:
        found = [
            c for c in expected_concepts
            if c.lower() in response_lower
        ]
        scores["concept_coverage"] = len(found) / len(expected_concepts)
        scores["concepts_found"] = found
        scores["concepts_missing"] = [
            c for c in expected_concepts
            if c.lower() not in response_lower
        ]
    else:
        scores["concept_coverage"] = None
        scores["concepts_found"] = []
        scores["concepts_missing"] = []

    # Score 2: Safety flag compliance
    required_flags = case.get("required_safety_flags", [])
    if required_flags:
        found_flags = [
            f for f in required_flags
            if f.lower() in response_lower
        ]
        scores["safety_compliance"] = (
            len(found_flags) / len(required_flags)
        )
        scores["flags_found"] = found_flags
        scores["flags_missing"] = [
            f for f in required_flags
            if f.lower() not in response_lower
        ]
    else:
        scores["safety_compliance"] = 1.0
        scores["flags_found"] = []
        scores["flags_missing"] = []

    # Score 3: Negative check
    must_not_contain = case.get("must_not_contain", [])
    violations = [
        m for m in must_not_contain
        if m.lower() in response_lower
    ]
    scores["negative_violations"] = violations
    scores["negative_check_passed"] = len(violations) == 0

    # Overall score
    component_scores = []
    if scores["concept_coverage"] is not None:
        component_scores.append(scores["concept_coverage"])
    if required_flags:
        component_scores.append(scores["safety_compliance"])
    if scores["negative_check_passed"]:
        component_scores.append(1.0)
    else:
        component_scores.append(0.0)

    scores["overall"] = (
        sum(component_scores) / len(component_scores)
        if component_scores else 0.0
    )

    return scores


def run_evaluation(provider: str = "claude",
                   model: str = "claude-sonnet-4-6",
                   output_dir: str = None) -> dict:
    """
    Run all test cases against the RAG pipeline and score results.
    Saves results to eval/results/ with a timestamp.
    """
    # Import here to avoid circular imports
    from rag.pipeline import RAGPipeline
    from llm.cost_tracker import CostTracker

    print(f"=== Health Worker AI Copilot — Evaluation Run ===")
    print(f"Provider: {provider} | Model: {model}")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    cases = load_test_cases()
    pipeline = RAGPipeline(
        provider=provider,
        model=model,
        cost_tracker=CostTracker()
    )

    results = []
    total_cost = 0.0

    for i, case in enumerate(cases, 1):
        print(f"Running TC{case['id'][-3:]} ({i}/{len(cases)}): "
              f"{case['category']} [{case['difficulty']}]")
        print(f"  Q: {case['question'][:80]}...")

        start_time = time.time()

        try:
            result = pipeline.query(case["question"])
            elapsed = time.time() - start_time
            scores = score_response(result["response"], case)
            total_cost += result["cost_usd"]

            print(f"  Overall score: {scores['overall']:.0%} | "
                  f"Time: {elapsed:.1f}s | "
                  f"Cost: ${result['cost_usd']:.4f}")

            if scores["concepts_missing"]:
                print(f"  Missing concepts: {scores['concepts_missing']}")
            if scores["flags_missing"]:
                print(f"  Missing safety flags: {scores['flags_missing']}")
            if scores["negative_violations"]:
                print(f"  VIOLATIONS: {scores['negative_violations']}")

            results.append({
                "case_id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "question": case["question"],
                "response": result["response"],
                "scores": scores,
                "input_tokens": result["input_tokens"],
                "output_tokens": result["output_tokens"],
                "cost_usd": result["cost_usd"],
                "elapsed_seconds": elapsed,
                "provider": provider,
                "model": model,
                "error": None
            })

        except Exception as e:
            elapsed = time.time() - start_time
            print(f"  ERROR: {str(e)}")
            results.append({
                "case_id": case["id"],
                "category": case["category"],
                "difficulty": case["difficulty"],
                "question": case["question"],
                "response": None,
                "scores": None,
                "error": str(e),
                "elapsed_seconds": elapsed,
                "provider": provider,
                "model": model
            })

        print()

    # Summary statistics
    scored = [r for r in results if r["scores"] is not None]
    avg_overall = (
        sum(r["scores"]["overall"] for r in scored) / len(scored)
        if scored else 0
    )
    avg_concept = (
        sum(r["scores"]["concept_coverage"] for r in scored
            if r["scores"]["concept_coverage"] is not None) /
        len([r for r in scored
             if r["scores"]["concept_coverage"] is not None])
        if scored else 0
    )
    avg_safety = (
        sum(r["scores"]["safety_compliance"] for r in scored) /
        len(scored) if scored else 0
    )

    summary = {
        "run_timestamp": datetime.now().isoformat(),
        "provider": provider,
        "model": model,
        "total_cases": len(cases),
        "successful_cases": len(scored),
        "failed_cases": len(cases) - len(scored),
        "avg_overall_score": round(avg_overall, 3),
        "avg_concept_coverage": round(avg_concept, 3),
        "avg_safety_compliance": round(avg_safety, 3),
        "total_cost_usd": round(total_cost, 4),
        "results": results
    }

    # Save results
    results_dir = (
        Path(output_dir) if output_dir
        else PROJECT_ROOT / "eval" / "results"
    )
    results_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"eval_{provider}_{timestamp}.json"

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print("=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Cases run:           {len(cases)}")
    print(f"Successful:          {len(scored)}")
    print(f"Average overall:     {avg_overall:.0%}")
    print(f"Concept coverage:    {avg_concept:.0%}")
    print(f"Safety compliance:   {avg_safety:.0%}")
    print(f"Total cost:          ${total_cost:.4f}")
    print(f"Ollama equivalent:   $0.0000")
    print(f"\nFull results saved to: {output_path}")

    return summary

#if __name__ == "__main__":
#    run_evaluation(
#        provider="ollama",
#        model="llama3.2"
#    )

if __name__ == "__main__":
    run_evaluation(
        provider="claude",
        model="claude-sonnet-4-6"
    )