import requests
import base64
import json
import os
from openai import OpenAI

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
prompt = "What is the derivative of x ** 2 + 5 * x + 3?"
stream = True


def get_api_key():
    api_key = os.getenv('SWISSAI_API')
    if not api_key:
        print("Error: SWISSAI_API environment variable not set")
        print("Please set your API key by running:")
        print("export SWISSAI_API='your-api-key'")
        print("You can find your API key at: http://148.187.108.173:8080/")
        exit(1)
    return api_key


def end_to_end(args):
    if args.multi_modal:
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ],
            }
        ]
    else:
        messages = [
            {
                "role": "user",
                "content": prompt,
            }
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
    if args.stream:
        for chunk in response:
            if len(chunk.choices) > 0 and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
    else:
        print(response.choices[0].message.content)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, default="meta-llama/Llama-3.2-90B-Vision-Instruct"
    )
    parser.add_argument("--stream", action="store_true", default=False)
    parser.add_argument("--constrained", action="store_true", default=False)
    parser.add_argument("--multi-modal", action="store_true", default=False)
    args = parser.parse_args()
    end_to_end(args)
