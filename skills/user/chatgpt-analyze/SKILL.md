# ChatGPT Analyze Skill

## When to Use
Use this skill when you need to analyze large texts using the OpenAI ChatGPT API. Ideal for:
- Summarizing long documents
- Extracting key insights from large bodies of text
- Pattern recognition in large text corpora
- Analyzing reports, articles, or any substantial text content

## Usage

```bash
python chatgpt_analyze.py --text "Your text here"
python chatgpt_analyze.py --file /path/to/document.txt
python chatgpt_analyze.py --file doc.txt --prompt "Summarize this in bullet points"
python chatgpt_analyze.py --file doc.txt --model gpt-4o --max-tokens 2000
```

## Input
- `--text`: Direct text content to analyze
- `--file`: Path to a text file to analyze
- `--prompt`: (Optional) Custom system prompt. Defaults to expert analyst prompt.
- `--model`: (Optional) OpenAI model name. Default: `gpt-4o`
- `--max-tokens`: (Optional) Maximum output tokens. Default: `4000`

## Output
ChatGPT's analysis printed to stdout as plain text.

## Required Environment Variable
- `OPENAI_API_KEY`: Your OpenAI API key. Set in environment or in a `.env` file.

## Large Text Handling
If the input text exceeds the model's context window, it is automatically chunked into parts, each analyzed separately, then a final summary is generated from the partial results.

## Dependencies
```
pip install openai python-dotenv
```
