import json
import os
import logging
import re

from llm_clients.DSV3Client import DSV3Client

# 设置logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def simple_sent_tokenize(text):
    """
    简单的句子分割函数，替代nltk.sent_tokenize
    """
    # 使用正则表达式分割句子
    sentences = re.split(r'[.!?]+\s+', text)
    # 清理空句子和只有空白字符的句子
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


def send_generate_request(prompt):
    llm_client = DSV3Client()
    return llm_client.chat(prompt)


def generate_user_summary(user_content):
    """
    生成用户内容的描述性摘要
    """
    prompt = f"""Role positioning: You are a professional conversation summarization expert who is good at analyzing and summarizing technical conversations.

Task description: Generate a concise and accurate descriptive summary based on the user's content, summarizing what the user is asking or requesting.

Constraints:
- The summary must be one sentence and no more than 80 characters
- It must accurately summarize the core content of the user's request
- Use third-person descriptions (e.g., "User asks...", "User requests...")
- Avoid the specifics of jargon and use plain language

Context:
{user_content[:800]}...

Workflow:
1. Understand the user's question or request
2. Distill the core content
3. Generate a one-sentence summary

Examples:
User: Please help me check this file
Summary: User asks for help checking a file

User: What's wrong with this code?
Summary: User asks about code problems

Now please generate a one-sentence summary for the above user content:"""

    try:
        summary = send_generate_request(prompt)
        # 清理摘要，确保是一句话
        summary = summary.strip()
        if summary.startswith("摘要：") or summary.startswith("Summary:"):
            summary = summary.split(":", 1)[1].strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating user summary: {e}")
        return "生成用户摘要失败"


def generate_assistant_summary(assistant_content):
    """
    生成助手内容的描述性摘要
    """
    prompt = f"""Role positioning: You are a professional conversation summarization expert who is good at analyzing and summarizing technical conversations.

Task description: Generate a concise and accurate descriptive summary based on the assistant's content, summarizing what the assistant is doing or responding.

Constraints:
- The summary must be one sentence and no more than 80 characters
- It must accurately summarize the core content of the assistant's response
- Use third-person descriptions (e.g., "Assistant analyzes...", "Assistant provides...")
- Avoid the specifics of jargon and use plain language

Context:
{assistant_content[:800]}...

Workflow:
1. Understand the assistant's response or actions
2. Distill the core content
3. Generate a one-sentence summary

Examples:
Assistant: I'll read the file and analyze it...
Summary: Assistant reads and analyzes the specified file

Assistant: I found syntax errors in the code...
Summary: Assistant identifies and analyzes syntax errors in the code

Now please generate a one-sentence summary for the above assistant content:"""

    try:
        summary = send_generate_request(prompt)
        # 清理摘要，确保是一句话
        summary = summary.strip()
        if summary.startswith("摘要：") or summary.startswith("Summary:"):
            summary = summary.split(":", 1)[1].strip()
        return summary
    except Exception as e:
        logger.error(f"Error generating assistant summary: {e}")
        return "生成助手摘要失败"


def convert_dialogue_to_receval(dialogue_data, is_first_round=True):
    """
    将对话格式数据转换为 ReCEval 所需的推理链格式
    """
    # 提取系统提示、用户问题和助手回复
    system_prompt = next((m['content'] for m in dialogue_data if m['role'] == 'system'), "")
    user_query = next((m['content'] for m in dialogue_data if m['role'] == 'user'), "")
    assistant_response = next((m['content'] for m in dialogue_data if m['role'] == 'assistant'), "")

    # 构建question
    if is_first_round:
        # 第一轮：system + user
        question = f"System: {system_prompt}\nUser: {user_query}"
        # 第一轮分别生成system+user摘要和assistant摘要
        user_context = f"System: {system_prompt}\nUser: {user_query}"
        user_summary = generate_user_summary(user_context)
        assistant_summary = generate_assistant_summary(assistant_response)
        # 用so连接
        combined_summary = f"{user_summary} So {assistant_summary}"
        reasoning_steps = [combined_summary]
    else:
        # 后续轮次：user
        question = f"User: {user_query}"
        # 分别生成user摘要和assistant摘要
        user_summary = generate_user_summary(user_query)
        assistant_summary = generate_assistant_summary(assistant_response)
        # 用so连接
        combined_summary = f"{user_summary} So {assistant_summary}"
        reasoning_steps = [combined_summary]

    # 构建推理步骤的依赖关系（线性依赖）
    steps = []
    for i in range(len(reasoning_steps)):
        if i == 0:
            continue  # 跳过第一个语句
        parents = list(range(i)) if i > 0 else []  # 依赖所有之前的步骤
        steps.append({"child": i, "parents": parents})

    # 构建符合 ReCEval 格式的数据
    return {
        "question": question,
        "sentences": {
            "perturbed": reasoning_steps,
            "hypothesis": reasoning_steps[-1] if reasoning_steps else ""
        },
        "steps": {
            "perturbed": steps
        },
        "perturbed": 0  # 默认未扰动
    }


def process_directory(input_file, output_dir):
    """
    处理目录中的所有 JSON 文件，转换为 ReCEval 格式
    每个instance_id生成一个JSON文件，将所有轮次的推理步骤拼接到perturbed中
    只保留第一个question和最后一个hypothesis
    """
    trajectory_entries = [json.loads(line) for line in open(input_file, 'r', encoding='utf-8')]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for trajectory_entry in trajectory_entries:
        instance_id = trajectory_entry['instance_id']
        trajectory_messages = trajectory_entry['messages']

        # 存储所有轮次的推理步骤
        all_reasoning_steps = []
        first_question = None

        idx = 0
        round_index = 0

        while idx < len(trajectory_messages):
            current_entry = []

            # 第一轮时加入 system 消息
            if round_index == 0 and trajectory_messages[idx]['role'] == 'system':
                current_entry.append(trajectory_messages[idx])
                idx += 1

            # user + assistant 对出现
            if idx + 1 < len(trajectory_messages) and \
                    trajectory_messages[idx]['role'] == 'user' and \
                    trajectory_messages[idx + 1]['role'] == 'assistant':

                current_entry.append(trajectory_messages[idx])
                current_entry.append(trajectory_messages[idx + 1])
                idx += 2

            else:
                if idx < len(trajectory_messages):
                    logger.error(f"Unmatched message pair at index {idx} for instance {instance_id}")
                break

            # 转换为 ReCEval 格式
            try:
                is_first_round = (round_index == 0)
                converted_data = convert_dialogue_to_receval(current_entry, is_first_round)

                # 保存第一个question
                if first_question is None:
                    first_question = converted_data['question']

                # 将推理步骤添加到总列表中
                all_reasoning_steps.extend(converted_data['sentences']['perturbed'])

                print(f"成功转换 {instance_id} 的第 {round_index} 轮对话")

            except Exception as e:
                print(f"处理 {instance_id} 的第 {round_index} 轮对话时出错: {str(e)}")
                continue

            round_index += 1

        # 生成最终的合并数据
        if all_reasoning_steps and first_question:
            # 构建推理步骤的依赖关系（线性依赖）
            steps = []
            for i in range(len(all_reasoning_steps)):
                if i == 0:
                    continue  # 跳过第一个语句
                parents = list(range(i)) if i > 0 else []  # 依赖所有之前的步骤
                steps.append({"child": i, "parents": parents})

            merged_data = {
                "question": first_question,
                "sentences": {
                    "perturbed": all_reasoning_steps,
                    "hypothesis": all_reasoning_steps[-1] if all_reasoning_steps else ""
                },
                "steps": {
                    "perturbed": steps
                },
                "perturbed": 0
            }

            # 保存合并后的数据
            output_file = os.path.join(output_dir, f"{instance_id}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)

            print(f"成功保存 {instance_id} 的合并数据到: {output_file} (共 {len(all_reasoning_steps)} 个推理步骤)")
        else:
            print(f"警告: {instance_id} 没有成功转换的数据")


if __name__ == "__main__":
    # 配置输入和输出路径
    trajectory_file = "./data/raw_data/deepseek-tools__deepseek-chat__t-0.00__p-1.00__c-0.00___swe_bench_lite_dev.jsonl"
    tree_files = "./data/perturbed_trees"

    # 处理整个目录
    process_directory(trajectory_file, tree_files)
