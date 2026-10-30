"""
Single Azure OpenAI client used by all agent nodes. Centralized here so the model,
temperature, and credentials are configured in exactly one place.
"""
from __future__ import annotations

import os

from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI

load_dotenv()

REQUIRED_VARS = ["AZURE_OPENAI_ENDPOINT", "AZURE_OPENAI_API_KEY",
                 "AZURE_OPENAI_DEPLOYMENT", "AZURE_OPENAI_API_VERSION"]
missing = [v for v in REQUIRED_VARS if not os.getenv(v)]
if missing:
    raise RuntimeError(f"Missing .env variables: {missing}")


def get_llm(temperature: float = 0.0) -> AzureChatOpenAI:
    """temperature=0.0 for narrative drafting -- this is a compliance document, not
    creative writing; low temperature keeps wording consistent and reduces the chance
    of the model inventing extra, uncited detail."""
    return AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        azure_deployment=os.environ["AZURE_OPENAI_DEPLOYMENT"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        temperature=temperature,
    )