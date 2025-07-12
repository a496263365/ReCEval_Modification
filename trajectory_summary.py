# trajectory_summary.py

import json
import os
import logging
from tqdm import tqdm

from llm_clients.DSV3Client import DSV3Client

# 设置logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

# 配置输入和输出路径
trajectory_file = "./data/raw_data/deepseek-tools__deepseek-chat__t-0.00__p-1.00__c-0.00___swe_bench_lite_dev.jsonl"
summary_dir = "./data/summary_data"


def generate_content_summary(type, content):
    if type == "env":
        prompt = f"""Role positioning: You are a professional conversation summarization expert, skilled in accurately summarizing both simple and complex technical conversations.

Task description: Given a message from user or tool (aka. environment) , generate a concise, faithful summary of the following user or tool message. For simple inputs, use a brief sentence. For longer or more complex inputs, you may use clauses (e.g., "because...", "that...", "which...") to preserve complex information within a single sentence, ensuring the core ideas, reasoning, and motivations are preserved.

Requirements:
- Be faithful to the original content; do not hallucinate or omit critical context.
- Use third-person descriptions (e.g., "User asks...", "Tool explains...")
- Preserve the logic and causal flow if the response includes reasoning or steps
- Avoid technical jargon where possible; use plain, readable language
- Do not overcompress — ensure key insights and background are not lost
- Prioritize clarity and informativeness over brevity.

Context:
{content}

Workflow:
1. Analyze the intent and structure of the message.
2. Extract key points, motivations, or actions.
3. Generate an accurate summary. Use 1 sentence for simple cases, and clauses for complex cases.

Examples:
User: Please help me check this file.
Summary: User asks for help reviewing a file.

User: I'm not sure whether the syntax error is due to indentation or missing colons. Can you help me figure it out?
Summary: User suspects a syntax error and seeks help identifying whether it's due to indentation or missing colons.

User: I tried running the script, but after importing pandas it crashes with a memory error. This only happens on large CSV files, not smaller ones.
Summary: User reports a memory error occurring only when using pandas with large CSV files.

Now please generate a summary for the above content based on its complexity and content."""
    elif type == "assistant":
        prompt = f"""Role positioning: You are a professional conversation summarization expert, skilled in accurately summarizing both simple and complex technical conversations.

Task description: Given a message from an assistant, generate a concise, faithful summary of the following user or tool message. For simple inputs, use a brief sentence. For longer or more complex inputs, you may use clauses (e.g., "because...", "that...", "which...") to preserve complex information within a single sentence, ensuring the core ideas, reasoning, and motivations are preserved.

Requirements:
- Be faithful to the original content; do not hallucinate or omit critical context.
- Use third-person descriptions (e.g., "Assistant explains...", "Assistant suggests...")
- Preserve the logic and causal flow if the response includes reasoning or steps
- Avoid technical jargon where possible; use plain, readable language
- Do not overcompress — ensure key insights and background are not lost
- Prioritize clarity and informativeness over brevity.

Context:
{content}

Workflow:
1. Analyze the intent and structure of the message.
2. Extract key points, motivations, or actions.
3. Generate an accurate summary. Use 1 sentence for simple cases, and clauses for complex cases.

Examples:

Assistant: I'll read the file and analyze it...
Summary: Assistant reads and analyzes the specified file.

Assistant: I found syntax errors in the code, likely due to indentation problems or missing colons. You should fix the structure and try again.
Summary: Assistant identifies syntax errors and suggests checking indentation and colons.

Assistant: Based on the error trace, it seems the API key is missing from the config file. You need to add your OpenAI key under the `api_key` field before rerunning.
Summary: Assistant diagnoses a missing API key and instructs the user to add it in the config file.

Now please generate a summary for the above assistant message, adjusting the length to its complexity.
"""
    else:
        logger.error("unknown type")
        return "错误的总结类型"

    llm_client = DSV3Client()
    try:
        summary = llm_client.chat(prompt)
        # 清理摘要，确保是一句话
        summary = summary.strip()
        if summary.startswith("摘要：") or summary.startswith("Summary:"):
            summary = summary.split(":", 1)[1].strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating user summary: {e}")
        return "生成用户摘要失败"


def generate_message_summary(message):
    role = message['role']
    if role == "user" or role == "tool":
        type = "env"
    elif role == "assistant":
        type = "assistant"
    else:
        logger.error("unknown role")

    content = f"{message['role']}: {message['content']}"
    if message.get("action") is not None:
        content += f"Then, I will take action: {message['action']} ."
    summary = generate_content_summary(type, content)
    summary_message = {
        'role': type,
        'content': summary,
    }
    return summary_message


def trajectory_2_summary(input_file, output_dir):
    """
    处理输入的 JSONL 格式文件，每个 instance_id 生成一个 JSON 文件
    为每段对话生成总结
    """

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    trajectory_entries = [json.loads(line) for line in open(input_file, 'r', encoding='utf-8')]

    for trajectory_entry in tqdm(trajectory_entries):
        instance_id = trajectory_entry['instance_id']
        trajectory_messages = trajectory_entry['messages']
        summary_messages = []

        # 检测格式问题并准备对话数据
        idx = 0
        while idx < len(trajectory_messages):
            # 合并第一轮 system 消息
            if idx == 0:
                if trajectory_messages[idx]['role'] == 'system':
                    idx += 1
                else:
                    logger.error(f"Trajectory {instance_id} 's first role isn't system")

            # user + assistant 对出现
            if (idx + 2 < len(trajectory_messages)
                    and (trajectory_messages[idx]['role'] == 'user' or trajectory_messages[idx]['role'] == 'tool')
                    and trajectory_messages[idx + 1]['role'] == 'assistant'):
                if idx == 1 and trajectory_messages[0]['role'] == 'system':
                    first_message = trajectory_messages[1]
                    first_message["content"] = trajectory_messages[0]["content"] + trajectory_messages[1]["content"]
                    summary_messages.append(generate_message_summary(first_message))
                    summary_messages.append(generate_message_summary(trajectory_messages[2]))
                    # pass
                else:
                    summary_messages.append(generate_message_summary(trajectory_messages[idx]))
                    summary_messages.append(generate_message_summary(trajectory_messages[idx + 1]))
                    # pass
                idx += 2

            elif idx == len(trajectory_messages) - 1:
                summary_messages.append(generate_message_summary(trajectory_messages[idx]))
                idx += 1
            else:
                if idx < len(trajectory_messages):
                    logger.error(f"Unmatched message pair at index {idx} for instance {instance_id}")
                break

        # 保存合并后的数据
        output_file = os.path.join(output_dir, f"{instance_id}.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary_messages, f, ensure_ascii=False, indent=2)
        logger.log(logging.INFO, f"成功保存 {instance_id} 的合并数据到: {output_file}")


if __name__ == "__main__":
    trajectory_2_summary(trajectory_file, summary_dir)
