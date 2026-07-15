import os
import yaml
from pathlib import Path
from anthropic import Anthropic
from dotenv import load_dotenv
from llm.cost_tracker import CostTracker

load_dotenv()


def load_config() -> dict:
    config_path = Path(__file__).parent.parent / "config" / "app_config.yaml"
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class LLMClient:
    """
    Abstraction layer for LLM providers.
    Currently supports: Claude (Anthropic)
    Planned: OpenAI, Ollama

    All application code calls this class.
    Nothing else imports from anthropic, openai, or ollama directly.
    Switching providers is always one config change.
    """

    def __init__(self,
                 provider: str = None,
                 model: str = None,
                 cost_tracker: CostTracker = None):
        config = load_config()
        self.provider = provider or config["default_provider"]
        self.model = model or config["default_model"]
        self.cost_tracker = cost_tracker or CostTracker()
        self._setup_client()

    def _setup_client(self):
        if self.provider == "claude":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        else:
            raise ValueError(
                f"Provider '{self.provider}' not yet implemented. "
                f"Supported: claude"
            )

    def complete(self,
                 system_prompt: str,
                 user_message: str,
                 max_tokens: int = None) -> dict:
        """
        Single method for all LLM calls.
        Returns response text, token usage, and cost.
        """
        config = load_config()
        max_tokens = max_tokens or config["max_tokens_per_response"]

        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens
            cost = self.cost_tracker.record(
                self.model,
                input_tokens,
                output_tokens
            )

            return {
                "text": response.content[0].text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "cost_usd": cost,
                "provider": self.provider,
                "model": self.model
            }


if __name__ == "__main__":
    client = LLMClient()

    response = client.complete(
        system_prompt="You are a clinical decision support assistant for TB treatment.",
        user_message="What is the standard first-line treatment for new TB patients?"
    )

    print(f"Response:\n{response['text']}\n")
    print(f"Tokens: {response['input_tokens']} in, {response['output_tokens']} out")
    print(f"This call cost: ${response['cost_usd']:.4f}")
    print(f"Session total: {client.cost_tracker.summary()['cost_display']}")