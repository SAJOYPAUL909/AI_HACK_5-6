from langchain_openai import ChatOpenAI

class LLMService:
    def __init__(self, api_key: str, base_url: str, model_name: str, temperature: float = 0):
        """
        Generic Service for ANY OpenAI-compatible API (Ollama, OpenRouter, vLLM, etc.)
        """
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo" if not model_name else model_name,
            openai_api_key=api_key if api_key else "EMPTY", # Ollama often needs dummy key
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=None,
            timeout=None,
            max_retries=2,
        )

    def get_llm(self):
        return self.llm