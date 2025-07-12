# receval_modification.py

import logging
from tqdm import tqdm
import os
import json
import re

from llm_clients.DSV3Client import DSV3Client

# 设置logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 配置输入和输出路径
summary_dir = "./data/summary_data"
result_dir = "./data/receval_result"

llm_client = DSV3Client()


def evaluate_intra_correctness(cur):
    prompt = f"""Role positioning: You are a professional dialogue quality evaluator specializing in assessing the internal coherence and correctness of conversation summaries.

Task description: Given a single-sentence summary of one round in a human-AI or tool-AI interaction, your task is to evaluate its **intra-correctness** — that is, whether this sentence is internally self-consistent, grammatically sound, and semantically coherent.

Requirements:
- Check if the sentence has **clear meaning**, **complete structure**, and **no logical contradictions**.
- Ensure there are **no hallucinated claims**, missing key context, or undefined references.
- Ignore fluency issues unless they affect comprehension.
- DO NOT evaluate informativeness or relation to other sentences.

Context (summary to evaluate):
{cur}

Scoring Instructions:
- Score the sentence from **0.0 to 1.0**, allowing any value in that range (e.g., 0.23, 0.62, 0.94).
- Think about the rubric below before deciding your score.

Scoring Rubric:
- 1.0: Perfectly correct and internally coherent.
- 0.75: Mostly coherent but has minor phrasing or clarity issues.
- 0.5: Contains moderate grammatical or logical problems.
- 0.25: Sentence is confusing, ambiguous, or missing critical parts.
- 0.0: Completely incoherent or nonsensical.

Important:
- Only output the final **float score**.
- Do not provide explanation, analysis, or extra text.
- The output must be a valid Python float (e.g., 0.67)

Now provide the score below:"""
    try:
        response = llm_client.chat(prompt)
        return get_score(response)
    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        return -1.0


def evaluate_inter_correctness(prev, cur):
    prompt = f"""Role positioning: You are a professional evaluator of dialogue reasoning chains, with expertise in assessing the logical and semantic coherence between consecutive conversation turns.

Task description: Given two single-sentence summaries of consecutive interaction rounds in a dialogue — the previous and the current — your task is to assess their **inter-correctness**, i.e., how well the current sentence logically and semantically follows from the previous one.

Requirements:
- Evaluate whether the current sentence is a natural and coherent continuation of the previous one.
- Look for consistency in topic, intent, and reference resolution.
- Penalize abrupt topic shifts, contradictions, or unexplained transitions.
- Do not judge internal sentence quality (that's intra-correctness).
- Do not assess informativeness or factual correctness — only continuity.

Context:
Previous sentence: {prev}
Current sentence: {cur}

Scoring Instructions:
- Provide a score from **0.0 to 1.0**, using floating-point numbers (e.g., 0.46, 0.83).
- Use the rubric below for guidance.

Scoring Rubric:
- 1.0: Perfectly coherent continuation; flows logically and clearly.
- 0.75: Mostly coherent; minor shifts or unclear references.
- 0.5: Somewhat disjointed or lacking logical clarity.
- 0.25: Largely incoherent or abrupt transition.
- 0.0: Completely unrelated or contradictory.

Important:
- Only output the final **float score**.
- Do not output explanation, comments, or anything besides the number.
- The output must be a valid Python float.

Now provide the score below:"""
    try:
        response = llm_client.chat(prompt)
        return get_score(response)
    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        return -1.0


def evaluate_informativeness(messages):
    history = messages[:-1]
    cur = messages[-1]
    history_str = "\n".join(history)
    prompt = f"""Role positioning: You are an expert judge of dialogue informativeness, with the ability to determine whether a statement meaningfully contributes new and relevant information.

Task description: Given a sentence from a dialogue, and the prior context (earlier dialogue turns), your task is to evaluate the **informativeness** of the current sentence — i.e., how much new, relevant, and helpful information it adds to the dialogue.

Requirements:
- Reward new insights, reasoning, actions, diagnoses, or clarifications.
- Penalize vague restatements, repetition, or unhelpful filler.
- Do not assess grammar or internal correctness (that’s intra-correctness).
- Do not judge its continuity (that’s inter-correctness).

Context:
Previous context: "{history_str}"
Current sentence: "{cur}"

Scoring Instructions:
- Provide a score from **0.0 to 1.0**, using floating-point numbers (e.g., 0.91, 0.33).
- Use the rubric below to guide your decision.

Scoring Rubric:
- 1.0: Adds clear, substantial, and novel information.
- 0.75: Adds somewhat useful or partially new information.
- 0.5: Contains limited or vague contributions.
- 0.25: Mostly redundant or off-topic.
- 0.0: No new information or entirely irrelevant.

Important:
- Only output the final **float score**.
- Do not output explanation, reasoning, or extra commentary.
- The output must be a valid Python float.

Now provide the score below:"""
    try:
        response = llm_client.chat(prompt)
        return get_score(response)
    except Exception as e:
        logger.error(f"Error in evaluate: {e}")
        return -1.0


def get_score(response):
    response = response.strip()
    if (response.startswith("得分：")
            or response.startswith("score:")
    ):
        response = response.split(":", 1)[1].strip()
    try:
        score = float(response)
        return score
    except Exception as e:
        match = re.search(r"\d+(?:\.\d+)?", response)
        if match:
            return float(match.group())
        else:
            logger.error(f"Error in transform response to score: {e}")
            return -1.0

def receval_summary(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    messages_content = [m['content'] for m in messages]

    intra_correctness_list = []
    inter_correctness_list = []
    informativeness_list = []

    for i in range(len(messages)):
        if i == 0:
            cur = messages_content[i]
            intra_correctness_list.append(evaluate_intra_correctness(cur))
        else:
            prev = messages_content[i - 1]
            cur = messages_content[i]
            intra_correctness_list.append(evaluate_intra_correctness(cur))
            inter_correctness_list.append(evaluate_inter_correctness(prev, cur))
            informativeness_list.append(evaluate_informativeness(messages_content[:i + 1]))

    receval_metrics = {
        'intra_correctness': intra_correctness_list,
        'inter_correctness': inter_correctness_list,
        'informativeness': informativeness_list,
        'min_intra_correctness': min(intra_correctness_list),
        'min_inter_correctness': min(inter_correctness_list),
        'min_informativeness': min(informativeness_list),
        'avg_intra_correctness': sum(intra_correctness_list) / len(intra_correctness_list),
        'avg_inter_correctness': sum(inter_correctness_list) / len(inter_correctness_list),
        'avg_informativeness': sum(informativeness_list) / len(informativeness_list),
    }
    return receval_metrics


def receval_predict(summary_dir, result_dir):
    for fname in tqdm(os.listdir(summary_dir)):
        if not fname.endswith('.json'):
            continue
        instance_id = fname[:-5]
        summary_path = os.path.join(summary_dir, fname)
        output_path = os.path.join(result_dir, f"{instance_id}.json")

        scores = receval_summary(summary_path)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(scores, f, indent=2, ensure_ascii=False)

        logger.log(logging.INFO, f"Completed ReCEval: {instance_id}")


if __name__ == '__main__':
    receval_predict(summary_dir, result_dir)
