import argparse
import json
import os.path
import time as time
import traceback
from functools import partial
from multiprocessing import Pool

import boto3
import pandas as pd
from botocore.config import Config
from tqdm import tqdm

from constant import SOAP_SECTIONS

CLAUDE_3_HAIKU_MODEL_ID = "anthropic.claude-3-haiku-20240307-v1:0"
CLAUDE_3_SONNET_MODEL_ID = "anthropic.claude-3-sonnet-20240229-v1:0"
MISTRAL_LARGE_V2_MODEL_ID = "mistral.mistral-large-2407-v1:0"
LLAMA_3_1_70B_MODEL_ID = "meta.llama3-1-70b-instruct-v1:0"

PROMPT_TEMPLATE_SOAP = """
In emotional support conversations, two primary roles exist: the therapist (individual providing support) and the client (individual seeking support). Your task is to summarize an emotional support conversation into client progress notes. These notes are usually in the SOAP format. The SOAP is a standardized form of recording a client's progress. It stands for:

- Subjective: In this section, document the subjective reports from the client, their family members, and past medical records. Include how the client describes their feelings and current symptoms.
- Objective: This section is for recording objective observations made during the session. Note any factual, observable information, such as the client's appearance, behavior, mood, affect, and speech patterns. Avoid including any subjective statements or self-reported information from the client. 
- Assessment: In this section, integrate the subjective and objective information to provide a comprehensive analysis of the client's current condition. Summarize your clinical impressions and hypotheses regarding the client's issues. 
- Plan: Outline the next steps for the client's treatment. Include both short-term and long-term goals, specifying what will be addressed in the next session as well as overall treatment objectives. Be clear and specific about your expectations and the client’s goals for the duration of treatment.

Output Dictionary template: 
{
"Subjective": "...",
"Objective": "...",
"Assessment": "...",
"Plan": "..."
}
Generate notes for the provided conversation in the above Dictionary style template. 
"""


def get_bedrock_session(region="us-west-2"):
    session = boto3.Session()

    config = Config(
        read_timeout=120,  # corresponds to inference time limit set for Bedrock
        connect_timeout=120,
        retries={
            "max_attempts": 5,
        },
    )

    bedrock = session.client(
        service_name="bedrock-runtime",
        region_name=region,
        endpoint_url=f"https://bedrock-runtime.{region}.amazonaws.com",
        config=config,
    )
    return bedrock


def predict_claude3(
    prompt, temperature=0.0, top_p=0.8, top_k=1, model_id="anthropic.claude-3-sonnet-20240229-v1:0"
):
    bedrock = get_bedrock_session()
    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": "",
    }

    body = {
        "max_tokens": 2048,
        "stop_sequences": ["\n\nHuman:"],
        "anthropic_version": "bedrock-2023-05-31",
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "messages": [{"role": "user", "content": prompt}],
    }

    api_template["body"] = json.dumps(body)

    success = False
    for i in range(5):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(1)

    if success:
        response_body = json.loads(response.get("body").read())
        return response_body["content"][0]["text"]
    else:
        print("***exception!!!!!")
        return ""


def predict_llama3(prompt, temperature=0.0, model_id=LLAMA_3_1_70B_MODEL_ID):
    bedrock = get_bedrock_session()
    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": "",
    }

    body = {
        "prompt": f"""
        <|begin_of_text|>
        <|start_header_id|>user<|end_header_id|>
        {prompt}
        <|eot_id|>
        <|start_header_id|>assistant<|end_header_id|>
        """,
        "temperature": temperature,
    }

    api_template["body"] = json.dumps(body)

    success = False
    for i in range(5):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(1)

    if success:
        response_body = json.loads(response.get("body").read())
        return response_body["generation"].strip()
    else:
        print("***exception!!!!!")
        return ""


def predict_mistral_v2(prompt, temperature=0.0, top_p=0.8, model_id=MISTRAL_LARGE_V2_MODEL_ID):
    bedrock = get_bedrock_session()
    api_template = {
        "modelId": model_id,
        "contentType": "application/json",
        "accept": "application/json",
        "body": "",
    }

    body = {
        "max_tokens": 2048,
        "temperature": temperature,
        "top_p": top_p,
        "messages": [{"role": "user", "content": prompt}],
    }

    api_template["body"] = json.dumps(body)

    success = False
    for i in range(10):
        try:
            response = bedrock.invoke_model(**api_template)
            success = True
            break
        except:
            traceback.print_exc()
            time.sleep(1)

    if success:
        response_body = json.loads(response.get("body").read())
        try:
            return response_body["choices"][0]["message"]["content"]
        except:
            print(response_body)
    else:
        print("***exception!!!!!")
        return ""


def parallel_apply(pair, function_name):
    convo_id, sub_df = pair
    temp_dict = {}
    transcript_str = "\n\n"
    for j in sub_df.itertuples():
        if j.interlocutor == "therapist":
            transcript_str = transcript_str + "\ntherapist: " + j.utterance_text
        else:
            transcript_str = transcript_str + "\nclient: " + j.utterance_text
    temp_dict["conversation"] = transcript_str

    prompt_str = PROMPT_TEMPLATE_SOAP + transcript_str + "\n\nSOAP Note:\n"
    soap_note = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
    for x in range(5):
        soap_note = function_name(prompt_str)
        try:
            soap_note = json.loads("{" + soap_note.split("{")[1].split("}")[0] + "}", strict=False)
            break
        except:
            soap_note = {"Subjective": "", "Objective": "", "Assessment": "", "Plan": ""}
            print(f"Exception in attempt {x}!")
            prompt_str += "\n\nGenerate notes for the provided conversation in the above Dictionary style template.\n"
            continue

    temp_dict["soap_note"] = {k.lower(): v for k, v in soap_note.items()}
    assert len(temp_dict["soap_note"]) == 4
    for section in temp_dict["soap_note"]:
        assert section in SOAP_SECTIONS
    return convo_id, temp_dict


def work_annomi(path, quality, function_name, name, output_filename="output_filename"):
    # Initialize lists and dictionary to store results
    outputs_annomi = {name: {}}

    # Read and filter the data
    data = pd.read_csv(path)
    data = data[data["mi_quality"] == quality]
    data = data.sort_values(by=["transcript_id", "timestamp"])
    data = data[["transcript_id", "utterance_text", "interlocutor"]]
    data = data.drop_duplicates()

    grouped_data = list(data.groupby("transcript_id"))

    with Pool(4) as p:
        outs = list(
            tqdm(
                p.imap(partial(parallel_apply, function_name=function_name), grouped_data),
                total=len(grouped_data),
            )
        )

    for convo_id, item in outs:
        if item is None:
            continue
        outputs_annomi[name][convo_id] = item

    with open(output_filename, "w") as fp:
        json.dump(outputs_annomi, fp)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="path to AnnoMI-full.csv")
    parser.add_argument("--output", type=str, help="outputs")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    function_and_model_name = [
        (partial(predict_claude3, model_id=CLAUDE_3_SONNET_MODEL_ID), "claude3sonnet"),
        (partial(predict_claude3, model_id=CLAUDE_3_HAIKU_MODEL_ID), "claude3haiku"),
        (partial(predict_llama3, model_id=LLAMA_3_1_70B_MODEL_ID), "llama31_70B"),
        (partial(predict_mistral_v2, model_id=MISTRAL_LARGE_V2_MODEL_ID), "mistral_large_v2"),
    ]
    quality_levels = ["low", "high"]

    # Iterate over function names and quality levels to process data
    for function_name, model_name in function_and_model_name:
        for quality in quality_levels:
            output_filename = f"{args.output}/outputs_annomi_{model_name}_{quality}.json"
            if os.path.exists(output_filename):
                print(f"SKIP: {output_filename}")
                continue
            print("------------")
            print(quality, model_name)
            work_annomi(
                path=args.input,
                quality=quality,
                function_name=function_name,
                name=model_name,
                output_filename=output_filename,
            )


if __name__ == "__main__":
    main()
