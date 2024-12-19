import requests
import base64
import json
import os
from openai import OpenAI
from datetime import datetime

os.environ["SWISSAI_API"] = "sk-rc-azOYl2hozNbsEh8oHcaMIA"

json_schema = json.dumps(
    {
        "type": "object",
        "properties": {
            "name": {"type": "string", "pattern": "^[\\w]+$"},
            "population": {"type": "integer"},
        },
        "required": ["name", "population"],
    }
)

def generate_paraphrasing_prompt(text):
    prompt = f"""
    You are an expert in rewriting text to ensure originality and avoid copyright infringement. 
    Your task is to carefully read and understand the provided text, then rewrite it in your own words. The rewritten version must:

    1. Faithfully convey the same meaning and ideas as the original text.
    2. Maintain the same tone and style as the source material.
    3. Use entirely new wording to create a unique and original version.

    Original Text:
    {text}

    Note: The provided text may be an excerpt from a larger article and could begin or end abruptly, even in the middle of a sentence. When rewriting, ensure that you preserve the original structure and flow without altering any abrupt starting or ending points.

    ONLY output the rewritten text:
    """
    return prompt


def get_api_key():
    api_key = os.getenv('SWISSAI_API')
    if not api_key:
        print("Error: SWISSAI_API environment variable not set")
        print("Please set your API key by running:")
        print("export SWISSAI_API='your-api-key'")
        print("You can find your API key at: http://148.187.108.173:8080/")
        exit(1)
    return api_key


def end_to_end(args, texts):

    prompts = [generate_paraphrasing_prompt(text) for text in texts]

    messages = [
        {
            "role": "user",
            "content": prompt,
        }

        for prompt in prompts
    ]

    client = OpenAI(
        base_url="https://fmapi.swissai.cscs.ch/",
        api_key=get_api_key(),
    )

    if args.constrained:
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "foo", "schema": json.loads(json_schema)},
        }
    else:
        response_format = None

    response = client.chat.completions.create(
        model=args.model,
        messages=messages,
        response_format=response_format,
        stream=args.stream,
    )

    # if args.stream:
    #     for chunk in response:
    #         if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
    #             print(chunk.choices[0].delta.content, end="", flush=True)
    # else:
    #     print(response.choices[0].message.content)

    paraphrases = [""] * len(texts)
    for p in response.choices:
        print(p.index)
        paraphrases[p.index] = response.choices[p.index].message.content
        print(paraphrases[p.index])


if __name__ == "__main__":
    import argparse
    from datasets import load_dataset
    from transformers import AutoTokenizer


    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-90B-Vision-Instruct"
    )
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--multi-modal", action="store_true", default=False)
    args = parser.parse_args()

    dataset = load_dataset(
        "json",
        data_files="/Users/xuyixuan/Downloads/Project/PDM/PDM/data/gutenberg_en_8k_token.jsonl",  
        split="train",
    )

    num_tokens = 100
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    texts = [tokenizer.decode(dataset[i]["input_ids"][:num_tokens], skip_special_tokens=True) for i in range(1)]

    print("Original Text:")
    print("---" * 20)
    for text in texts:
        print(text)
        print("---" * 20)
    print("\n")
    print("---" * 20)
    end_to_end(args, texts)
    print("---" * 20)

