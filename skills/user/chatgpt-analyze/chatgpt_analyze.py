#!/usr/bin/env python3
"""Analyze large texts using the OpenAI ChatGPT API."""

import argparse
import os
import sys
import time

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from openai import OpenAI, RateLimitError, APITimeoutError, APIConnectionError
except ImportError:
    print("Error: openai package not installed. Run: pip install openai", file=sys.stderr)
    sys.exit(1)

DEFAULT_SYSTEM_PROMPT = (
    "You are an expert analyst. Analyze the provided text thoroughly. "
    "Extract key insights, patterns, and important information. "
    "Structure your response clearly."
)

# Approximate characters per token (conservative estimate)
CHARS_PER_TOKEN = 3
# Leave headroom for system prompt and response
MAX_INPUT_CHARS = {
    "gpt-4o": 400_000,       # ~128k tokens context
    "gpt-4o-mini": 400_000,
    "gpt-4-turbo": 400_000,
    "gpt-4": 24_000,         # ~8k tokens context
    "gpt-3.5-turbo": 12_000, # ~4k tokens context
}
DEFAULT_MAX_CHARS = 400_000


def get_max_chars(model: str) -> int:
    for key in MAX_INPUT_CHARS:
        if model.startswith(key):
            return MAX_INPUT_CHARS[key]
    return DEFAULT_MAX_CHARS


def chunk_text(text: str, max_chars: int) -> list[str]:
    """Split text into chunks that fit within the context limit."""
    if len(text) <= max_chars:
        return [text]
    chunks = []
    while text:
        chunk = text[:max_chars]
        # Try to break at a paragraph or sentence boundary
        for sep in ["\n\n", "\n", ". ", " "]:
            idx = chunk.rfind(sep)
            if idx > max_chars // 2:
                chunk = text[: idx + len(sep)]
                break
        chunks.append(chunk.strip())
        text = text[len(chunk):]
    return chunks


def call_api(client: OpenAI, model: str, system_prompt: str, user_content: str, max_tokens: int) -> str:
    """Call the OpenAI Chat Completions API with retry logic."""
    delays = [2, 4, 8, 16]
    last_error = None
    for attempt, delay in enumerate([0] + delays):
        if delay:
            time.sleep(delay)
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except RateLimitError as e:
            last_error = e
            if attempt < len(delays):
                print(f"Rate limit hit, retrying in {delays[attempt]}s...", file=sys.stderr)
            else:
                print("Error: Rate limit exceeded after retries.", file=sys.stderr)
                sys.exit(1)
        except APITimeoutError as e:
            last_error = e
            if attempt < len(delays):
                print(f"Request timed out, retrying in {delays[attempt]}s...", file=sys.stderr)
            else:
                print("Error: Request timed out after retries.", file=sys.stderr)
                sys.exit(1)
        except APIConnectionError as e:
            last_error = e
            if attempt < len(delays):
                print(f"Connection error, retrying in {delays[attempt]}s...", file=sys.stderr)
            else:
                print("Error: Connection failed after retries.", file=sys.stderr)
                sys.exit(1)
    raise last_error


def analyze(text: str, system_prompt: str, model: str, max_tokens: int) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY environment variable is not set.", file=sys.stderr)
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    max_chars = get_max_chars(model)
    chunks = chunk_text(text, max_chars)

    if len(chunks) == 1:
        return call_api(client, model, system_prompt, chunks[0], max_tokens)

    # Analyze each chunk separately, then summarize
    print(f"Text is large, splitting into {len(chunks)} chunks...", file=sys.stderr)
    partial_results = []
    for i, chunk in enumerate(chunks, 1):
        print(f"Analyzing chunk {i}/{len(chunks)}...", file=sys.stderr)
        result = call_api(
            client, model,
            system_prompt,
            f"[Part {i} of {len(chunks)}]\n\n{chunk}",
            max_tokens,
        )
        partial_results.append(f"=== Part {i} Analysis ===\n{result}")

    print("Generating final summary...", file=sys.stderr)
    combined = "\n\n".join(partial_results)
    summary_prompt = (
        "You are an expert analyst. The following are analyses of different parts of a large document. "
        "Synthesize them into a single coherent analysis, merging insights and removing redundancy."
    )
    return call_api(client, model, summary_prompt, combined, max_tokens)


def main():
    parser = argparse.ArgumentParser(description="Analyze text using the OpenAI ChatGPT API.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", help="Text content to analyze")
    group.add_argument("--file", help="Path to a text file to analyze")
    parser.add_argument("--prompt", default=DEFAULT_SYSTEM_PROMPT, help="System prompt")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model name (default: gpt-4o)")
    parser.add_argument("--max-tokens", type=int, default=4000, help="Max output tokens (default: 4000)")
    args = parser.parse_args()

    if args.text:
        text = args.text
    else:
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                text = f.read()
        except FileNotFoundError:
            print(f"Error: File not found: {args.file}", file=sys.stderr)
            sys.exit(1)
        except IOError as e:
            print(f"Error reading file: {e}", file=sys.stderr)
            sys.exit(1)

    text = text.strip()
    if not text:
        print("Error: Input text is empty.", file=sys.stderr)
        sys.exit(1)

    result = analyze(text, args.prompt, args.model, args.max_tokens)
    print(result)


if __name__ == "__main__":
    main()
