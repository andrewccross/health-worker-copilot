import yaml
from pathlib import Path


class CostTracker:
    """
    Tracks token usage and estimates API costs per session.

    Design decision: cost tracking is separate from the LLM client
    because it needs to accumulate across multiple calls and be
    readable by the UI layer independently of the LLM layer.

    Ollama costs are always $0.00 — this is the point.
    Displaying this alongside API costs makes the sovereignty
    argument concrete rather than abstract.
    """

    def __init__(self):
        self.pricing = self._load_pricing()
        self.reset()

    def _load_pricing(self) -> dict:
        pricing_path = Path(__file__).parent.parent / "config" / "pricing.yaml"
        with open(pricing_path, "r") as f:
            return yaml.safe_load(f)

    def reset(self):
        """Call this at the start of each user session."""
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost_usd = 0.0
        self.query_count = 0
        self.calls = []

    def record(self, model: str, input_tokens: int, output_tokens: int):
        """
        Record a single LLM call.
        Called by LLMClient.complete() automatically.
        """
        cost = self._calculate_cost(model, input_tokens, output_tokens)

        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        self.total_cost_usd += cost
        self.query_count += 1

        self.calls.append({
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost
        })

        return cost

    def _calculate_cost(self,
                        model: str,
                        input_tokens: int,
                        output_tokens: int) -> float:
        if model not in self.pricing:
            return 0.0

        rates = self.pricing[model]
        input_cost = (input_tokens / 1_000_000) * rates["input_per_million"]
        output_cost = (output_tokens / 1_000_000) * rates["output_per_million"]
        return input_cost + output_cost

    def summary(self) -> dict:
        """Returns a dict the Streamlit sidebar can display directly."""
        return {
            "queries": self.query_count,
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "total_tokens": self.total_input_tokens + self.total_output_tokens,
            "cost_usd": round(self.total_cost_usd, 4),
            "cost_display": f"${self.total_cost_usd:.4f}"
        }


if __name__ == "__main__":
    tracker = CostTracker()

    # Simulate two API calls
    tracker.record("claude-sonnet-4-6", input_tokens=34, output_tokens=432)
    tracker.record("claude-sonnet-4-6", input_tokens=1800, output_tokens=380)

    summary = tracker.summary()
    print(f"Queries:       {summary['queries']}")
    print(f"Total tokens:  {summary['total_tokens']}")
    print(f"Estimated cost: {summary['cost_display']}")
    print(f"Ollama equivalent: $0.0000")