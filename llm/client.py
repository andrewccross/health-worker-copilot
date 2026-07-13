import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

class LLMClient:
    """
    Abstraction layer for LLM providers.
    Currently supports: Claude (Anthropic)
    Planned: OpenAI, Ollama
    
    All application code calls this class.
    Nothing else imports from anthropic, openai, or ollama directly.
    This means switching providers is always one config change.
    """

    def __init__(self, provider: str = "claude", model: str = None):
        self.provider = provider
        self.model = model
        self._setup_client()

    def _setup_client(self):
        if self.provider == "claude":
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
            if not self.model:
                self.model = "claude-sonnet-4-6"
        else:
            raise ValueError(f"Provider '{self.provider}' not yet implemented. "
                           f"Supported: claude")

    def complete(self, 
                 system_prompt: str, 
                 user_message: str,
                 max_tokens: int = 1024) -> dict:
        """
        Single method for all LLM calls.
        Returns a dict with response text and token usage.
        Token usage feeds the cost tracker in Stage 2.
        """
        if self.provider == "claude":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )
            return {
                "text": response.content[0].text,
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
                "provider": self.provider,
                "model": self.model
            }

# Simple test — run this file directly to verify your API key works
if __name__ == "__main__":
    client = LLMClient(provider="claude")
    
    response = client.complete(
        system_prompt="You are a clinical decision support assistant for TB treatment.",
        user_message="What is the standard first-line treatment regimen for new TB patients?"
    )
    
    print(f"Response: {response['text']}")
    print(f"Tokens used: {response['input_tokens']} in, {response['output_tokens']} out")