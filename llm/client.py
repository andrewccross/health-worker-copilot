import os
import yaml
from pathlib import Path
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
    Supports: Claude (Anthropic), OpenAI, Ollama (local)

    Design decision: provider imports are deferred to _setup_client()
    rather than imported at the top of the file. This means the app
    doesn't crash on startup if a provider's package isn't installed
    or its API key is missing — it only fails when that provider is
    actually selected. This is important for a tool where some
    deployment contexts (Rung 4) will never have API keys at all.
    """

    def __init__(self,
                 provider: str = None,
                 model: str = None,
                 cost_tracker: CostTracker = None):
        config = load_config()
        self.provider = provider or config["default_provider"]
        self.model = model or self._default_model()
        self.cost_tracker = cost_tracker or CostTracker()
        self._setup_client()

    def _default_model(self) -> str:
        defaults = {
            "claude": "claude-sonnet-4-6",
            "openai": "gpt-4o",
            "ollama": "llama3.2"
        }
        return defaults.get(self.provider, "claude-sonnet-4-6")

    def _setup_client(self):
        if self.provider == "claude":
            from anthropic import Anthropic
            self.client = Anthropic(
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )

        elif self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(
                api_key=os.getenv("OPENAI_API_KEY")
            )

        elif self.provider == "ollama":
            import ollama as ollama_sdk
            self.client = ollama_sdk
            # Ollama runs locally — no API key needed
            # Verify connection
            try:
                self.client.list()
            except Exception:
                raise ConnectionError(
                    "Cannot connect to Ollama. "
                    "Make sure Ollama is running: 'ollama serve'"
                )
        else:
            raise ValueError(
                f"Provider '{self.provider}' not supported. "
                f"Choose: claude, openai, ollama"
            )

    def complete(self,
                 system_prompt: str,
                 user_message: str,
                 max_tokens: int = None) -> dict:
        config = load_config()
        max_tokens = max_tokens or config["max_tokens_per_response"]

        if self.provider == "claude":
            return self._complete_claude(
                system_prompt, user_message, max_tokens
            )
        elif self.provider == "openai":
            return self._complete_openai(
                system_prompt, user_message, max_tokens
            )
        elif self.provider == "ollama":
            return self._complete_ollama(
                system_prompt, user_message, max_tokens
            )

    def _complete_claude(self, system_prompt, user_message, max_tokens):
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}]
        )

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = self.cost_tracker.record(
            self.model, input_tokens, output_tokens
        )

        return {
            "text": response.content[0].text,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "provider": self.provider,
            "model": self.model
        }

    def _complete_openai(self, system_prompt, user_message, max_tokens):
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        input_tokens = response.usage.prompt_tokens
        output_tokens = response.usage.completion_tokens
        cost = self.cost_tracker.record(
            self.model, input_tokens, output_tokens
        )

        return {
            "text": response.choices[0].message.content,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": cost,
            "provider": self.provider,
            "model": self.model
        }

    def _complete_ollama(self, system_prompt, user_message, max_tokens):
        response = self.client.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ]
        )

        # Ollama doesn't return token counts consistently
        # Estimate from character count for cost display purposes
        input_text = system_prompt + user_message
        input_tokens = len(input_text) // 4
        output_tokens = len(
            response["message"]["content"]
        ) // 4

        # Cost is always zero for Ollama
        self.cost_tracker.record("ollama", input_tokens, output_tokens)

        return {
            "text": response["message"]["content"],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost_usd": 0.0,
            "provider": self.provider,
            "model": self.model
        }


if __name__ == "__main__":
    # Test Claude only — OpenAI and Ollama tested via the UI
    client = LLMClient(provider="claude")
    response = client.complete(
        system_prompt="You are a TB clinical decision support assistant.",
        user_message="What is first-line TB treatment?"
    )
    print(f"Response: {response['text'][:200]}...")
    print(f"Cost: ${response['cost_usd']:.4f}")