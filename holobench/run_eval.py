import asyncio
import json
from pathlib import Path
from typing import List, Literal

import fire
import openai
import pandas as pd
from pydantic import BaseModel
from tqdm.asyncio import tqdm


class ParsedResponse(BaseModel):
    reasoning: str
    final_answer: str


class Eval(BaseModel):
    id: int
    label: Literal["Exact Match", "Partial Match", "No Match"]


class EvaluationResponse(BaseModel):
    evaluations: List[Eval]


async def proc(
    question: str,
    prediction: str,
    gold: List[dict],
    eval_template: str,
    post_proc_template: str,
    client: openai.OpenAI,
    model_name: str = "gpt-4o-mini-2024-07-18",
    k=50,
):
    """
    Processes a single row of the input data. This function handles both post-processing of the model's response and evaluation of
    the prediction against gold standard data.

    Args:
    - question (str): The question posed to the model.
    - prediction (str): The model's prediction.
    - gold (List[dict]): List of gold standard answers in dictionary format.
    - eval_template (str): Template used for the evaluation prompt.
    - post_proc_template (str): Template used for post-processing the prediction.
    - client (openai.OpenAI): The OpenAI client object for making API calls.
    - model_name (str): Name of the OpenAI model to use. Default is "gpt-4o-mini-2024-07-18".
    - k (int): The maximum number of gold examples to process. Default is 50.

    Returns:
    - dict: A dictionary containing the question, prediction, gold data, evaluations, reasoning, and final answer.
    """
    content = post_proc_template.format(response=prediction)
    response = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        response_format=ParsedResponse,
        temperature=0.0,
    )
    parsed = response.choices[0].message.parsed.model_dump()
    gold = [{"id": i} | x for i, x in enumerate(gold[:k], 1)]
    output_format = (
        [{"id": x["id"], "label": "..."} for x in gold]
        if gold
        else [{"id": 0, "label": "..."}]
    )
    content = eval_template.format(
        question=question,
        pred=parsed["final_answer"],
        gold=gold,
        output_format=output_format,
    )

    response = await client.beta.chat.completions.parse(
        model=model_name,
        messages=[{"role": "user", "content": content}],
        response_format=EvaluationResponse,
        temperature=0.0,
    )
    evals = response.choices[0].message.parsed.model_dump()["evaluations"]
    return {
        "question": question,
        "prediction": prediction,
        "gold": gold,
        "evaluations": evals,
        "reasoning": parsed["reasoning"],
        "final_answer": parsed["final_answer"],
    }


async def run_async(
    pred_file: str,
    output_dir: str,
    eval_prompt_file: str,
    post_process_prompt_file: str,
    model_name: str = "gpt-4o-mini-2024-07-18",
    api_key: str = None,
    base_url: str = None,
):
    """
    Main asynchronous function that orchestrates the processing of each row in the input file.

    Args:
    - pred_file (str): Path to the CSV file containing predictions.
    - output_dir (str): Directory where the output JSON file will be saved.
    - eval_prompt_file (str): Path to the evaluation prompt file.
    - post_process_prompt_file (str): Path to the post-processing prompt file.
    - model_name (str): Name of the OpenAI model to use. Default is "gpt-4o-mini-2024-07-18".
    - api_key (str): OpenAI API key. Default is None.
    - base_url (str): Optional base URL for API requests. Default is None.

    """
    pred_file = Path(pred_file)
    output_file = Path(output_dir) / pred_file.name
    df = pd.read_csv(pred_file)
    eval_template = open(eval_prompt_file).read()
    post_proc_template = open(post_process_prompt_file).read()
    client = openai.AsyncOpenAI(api_key=api_key, base_url=base_url)

    tasks = [
        proc(
            question=row.question,
            prediction=row.prediction,
            gold=json.loads(row.gold),
            eval_template=eval_template,
            post_proc_template=post_proc_template,
            client=client,
            model_name=model_name,
        )
        for _, row in df.iterrows()
    ]

    results = await tqdm.gather(*tasks)

    pd.DataFrame(results).to_json(output_file, orient="records")


def run(
    pred_file: str,
    output_dir: str,
    eval_prompt_file: str,
    post_process_prompt_file: str,
    model_name: str = "gpt-4o-mini-2024-07-18",
    api_key: str = None,
    base_url: str = None,
):
    asyncio.run(
        run_async(
            pred_file,
            output_dir,
            eval_prompt_file,
            post_process_prompt_file,
            model_name,
            api_key,
            base_url,
        )
    )


if __name__ == "__main__":
    fire.Fire(run)
