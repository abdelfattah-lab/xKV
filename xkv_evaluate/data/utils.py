################################################################################
#
# Copyright 2024 ByteDance Ltd. and/or its affiliates. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
################################################################################
# Some code comes from experments/ in MInference
# Original license:
# Copyright (c) Microsoft Corporation. and affiliates All rights reserved.
#
# See LICENSE.txt for license information
################################################################################

import re
import json
import random
import torch

def truncate_input(input: torch.LongTensor, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if input.shape[-1] <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return torch.cat([input[:, 0:split],input[:, -split:]], dim=-1)
    else:
        return None

def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input, return_tensors="pt")
    len_before = tokens.shape[-1]
    # print(f"# tokens before: {len_before} (max_tokens: {max_tokens})")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = tokens.shape[-1]  # type: ignore
    # print(f"# tokens after: {len_after} (max_tokens: {max_tokens})")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens

########## NIAH ##########

NIAH_TEMPLATE = "Write a high-quality answer for the given question using only the provided search results (some of which might be irrelevant).\n{context}\n\nQuestion: {question} Don't give information outside the document or repeat your findings. Keep your response short and direct. Answer: "

RANDOM_NEEDLE_CITIES = ["Chicago", "Yangon", "Antananarivo", "Colombo", "Almaty", "Sydney", "Chicago", "Mexico City", "Seattle", "Lagos", "Amsterdam", "Belgrade", "Cairo", "Baghdad", "Damascus", "Kigali", "Dakar", "Dakar", "Sofia", "Kigali", "Victoria", "Tashkent", "Mumbai", "Barcelona", "Almaty", "Amman", "Toronto", "Bratislava", "Johannesburg", "Thimphu", "Bangkok", "Santiago", "Cairo", "San Francisco", "Lagos", "Amsterdam", "Paris", "Rabat", "Santiago", "Copenhagen", "Madrid", "Kigali", "Ho Chi Minh City", "Sarajevo", "Delhi", "Istanbul", "Ho Chi Minh City", "Khartoum", "Helsinki", "Doha", "Istanbul", "Kuala Lumpur", "Budapest", "Shanghai", "Moscow", "Los Angeles", "Oslo", "Johannesburg", "Berlin", "Bangalore", "Tokyo", "Melbourne", "Barcelona", "Chicago", "Port Louis", "Lisbon", "Nairobi", "Kampala", "Lima", "Maputo", "Vancouver", "Dubai", "Khartoum", "Jakarta", "Madrid", "Yerevan", "Beirut", "Athens", "Chicago", "Paris", "Bucharest", "Copenhagen", "Brussels", "Damascus", "Seattle", "Los Angeles", "Yerevan", "Victoria", "Tunis", "Astana", "Seoul", "Buenos Aires", "Bangkok", "Colombo", "Brussels", "Khartoum", "Doha", "San Francisco", "Vienna", "Jakarta"]

def generate_random_number(num_digits):
    lower_bound = 10 ** (num_digits - 1)
    upper_bound = 10**num_digits - 1
    return random.randint(lower_bound, upper_bound)

def read_context_files(n, context_lengths, haystack_file, tokenizer):
    max_context_length = max(context_lengths)
    contexts = []
    f = open(haystack_file, "r")
    for _ in range(n):
        context = ""
        toks = 0
        while toks < max_context_length:
            text = json.loads(f.readline())["text"]
            context += text
            toks += len(tokenizer.encode(text))
        contexts.append(context)
    return contexts

def insert_needle_func(needle, context, depth_percent, context_length, tokenizer, final_context_length_buffer):
    tokens_needle = tokenizer.encode(needle, add_special_tokens=False)
    tokens_context = tokenizer.encode(context, add_special_tokens=False)

    # Reducing the context length by 150 buffer. This is to account for system message, the user question, and response.
    context_length -= final_context_length_buffer

    # If your context + needle are longer than the context length (which it will be), then reduce tokens from the context by the needle length
    if len(tokens_context) + len(tokens_needle) > context_length:
        tokens_context = tokens_context[: context_length - len(tokens_needle)]

    if depth_percent == 100:
        # If your depth percent is 100 (which means your needle is the last thing in the doc), throw it at the end
        tokens_new_context = tokens_context + tokens_needle
    else:
        # Go get the position (in terms of tokens) to insert your needle
        insertion_point = int(len(tokens_context) * (depth_percent / 100))

        # tokens_new_context represents the tokens before the needle
        tokens_new_context = tokens_context[:insertion_point]

        # We want to make sure that we place our needle at a sentence break so we first see what token a '.' is
        period_tokens = [tokenizer.encode(".", add_special_tokens=False)[0], tokenizer.encode(". \n", add_special_tokens=False)[0], tokenizer.encode(".\n", add_special_tokens=False)[0], tokenizer.encode("\n", add_special_tokens=False)[0]]

        # Then we iteration backwards until we find the first period
        while tokens_new_context and tokens_new_context[-1] not in period_tokens:
            insertion_point -= 1
            tokens_new_context = tokens_context[:insertion_point]

        # Once we get there, then add in your needle, and stick the rest of your context in on the other end.
        # Now we have a needle in a haystack
        tokens_new_context += tokens_needle + tokens_context[insertion_point:]

    # Convert back to a string and return it
    new_context = tokenizer.decode(tokens_new_context, skip_special_tokens=True)
    return new_context

def create_contexts(
    needle_rnd_number,
    insert_needle,
    random_city,
    trim_context,
    context_length,
    depth_percent,
    needle,
    retrieval_question,
    tokenizer,
    final_context_length_buffer,
):
    needle = needle.format(city=random_city, rnd_number=needle_rnd_number)
    question = retrieval_question.format(random_city)
    if not insert_needle:
        needle = " "  # replace needle with a space
    context = insert_needle_func(
        needle, trim_context, depth_percent, context_length, tokenizer, final_context_length_buffer
    )
    results = {
        "context": context,
        "context_length": int(context_length),
        "depth_percent": float(depth_percent),
        "needle": needle,
        "question": question,
        "insert_needle": insert_needle,
        "needle_rnd_number": needle_rnd_number,
    }
    return results

########## LONG BENCH ##########

LONG_BENCH_TEMPLATE = {
    "narrativeqa": "You are given a story, which can be either a novel or a movie script, and a question. Answer the question asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nStory: {context}\n\nNow, answer the question based on the story asconcisely as you can, using a single phrase if possible. Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "qasper": "You are given a scientific article and a question. Answer the question as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nArticle: {context}\n\n Answer the question based on the above article as concisely as you can, using a single phrase or sentence if possible. If the question cannot be answered based on the information in the article, write \"unanswerable\". If the question is a yes/no question, answer \"yes\", \"no\", or \"unanswerable\". Do not provide any explanation.\n\nQuestion: {input}\n\nAnswer:",
    "multifieldqa_en": "Read the following text and answer briefly.\n\n{context}\n\nNow, answer the following question based on the above text, only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "multifieldqa_zh": "阅读以下文字并用中文简短回答：\n\n{context}\n\n现在请基于上面的文章回答下面的问题，只告诉我答案，不要输出任何其他字词。\n请用中文回答。\n问题：{input}\n回答：",
    "hotpotqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "2wikimqa": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "musique": "Answer the question based on the given passages. Only give me the answer and do not output any other words.\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:",
    "dureader": "请基于给定的文章回答下述问题。\n\n文章：{context}\n\n请基于上述文章回答下面的问题。\n请用中文回答。\n问题：{input}\n回答：",
    "gov_report": "You are given a report by a government agency. Write a one-page summary of the report.\n\nReport:\n{context}\n\nNow, write a one-page summary of the report.\n\nSummary:",
    "qmsum": "You are given a meeting transcript and a query containing a question or instruction. Answer the query in one or more sentences.\n\nTranscript:\n{context}\n\nNow, answer the query based on the above meeting transcript in one or more sentences.\n\nQuery: {input}\nAnswer:",
    "multi_news": "You are given several news passages. Write a one-page summary of all news. \n\nNews:\n{context}\n\nNow, write a one-page summary of all the news.\n\nSummary:",
    "vcsum": "下面有一段会议记录，请你阅读后，写一段总结，总结会议的内容。\n请用中文回答。\n会议记录：\n{context}\n\n会议总结：",
    "trec": "Please determine the type of the question below. Here are some examples of questions.\n\n{context}\n{input}",
    "triviaqa": "Answer the question based on the given passage. Only give me the answer and do not output any other words. The following are some examples.\n\n{context}\n\n{input}",
    "samsum": "Summarize the dialogue into a few short sentences. The following are some examples.\n\n{context}\n\n{input}",
    "lsht": "请判断给定新闻的类别，下面是一些例子。\n请用中文回答。\n{context}\n{input}",
    "passage_count": "There are some paragraphs below sourced from Wikipedia. Some of them may be duplicates. Please carefully read these paragraphs and determine how many unique paragraphs there are after removing duplicates. In other words, how many non-repeating paragraphs are there in total?\n\n{context}\n\nPlease enter the final count of unique paragraphs after removing duplicates. The output format should only contain the number, such as 1, 2, 3, and so on.\n\nThe final answer is: ",
    "passage_retrieval_en": "Here are 30 paragraphs from Wikipedia, along with an abstract. Please determine which paragraph the abstract is from.\n\n{context}\n\nThe following is an abstract.\n\n{input}\n\nPlease enter the number of the paragraph that the abstract is from. The answer format must be like \"Paragraph 1\", \"Paragraph 2\", etc.\n\nThe answer is: ",
    "passage_retrieval_zh": "以下是若干段落文字，以及其中一个段落的摘要。请确定给定的摘要出自哪一段。\n\n{context}\n\n下面是一个摘要\n\n{input}\n\n请输入摘要所属段落的编号。请用中文回答。答案格式必须是\"段落1\"，\"段落2\"等格式\n\n答案是：",
    "lcc": "Please complete the code given below. \n{context}Next line of code:\n",
    "repobench-p": "Please complete the code given below. \n{context}{input}Next line of code:\n"
}

########## INFINI BENCH ##########

VANILLA_INFINI_BENCH_TEMPLATE = {
    "passkey": "There is an important info hidden inside a lot of irrelevant text. Find it and memorize it. I will quiz you about the important information.\n\n{context}\n\n{input}\n\nThe pass key is",
    
    "number_string": "There is an important info hidden inside a lot of irrelevant text. Find it. I will quiz you about the important information there.\n\n{context}\n\n{input}\n\nThe sequence of digits is",
    
    "kv_retrieval": "Extract the value corresponding to the specified key in the JSON object below. A specified key value pair is hidden within the following text. Make sure to memorize it. I will quiz you about the key value pair afterwards.\n\n{context}\n\nWhat is the specified value for '{input}' mentioned in the provided JSON? Please do not reply with the key, but with the value corresponding to the key.The value associated with '{input}' is:",

    "longbook_sum_eng": "Summarize the book below. \n\n{context}\n\nSummary:",

    "longbook_choice_eng": "Read the book and answer the question.\n\n{context}\n\nQuestion: {question}\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe letter of the correct answer is",
    
    "longbook_qa_eng": "Read the book and answer the question. Be very concise in your answer.\n\n{context}\n\nQuestion: {question}\nAnswer:",
    
    "longbook_qa_chn": "阅读以下书籍然后回答问题。\n\n{context}\n请用中文回答。\n问题：{question}\n答案：",
    
    "math_find": "{prefix}\n\n{context}\n\n{input}",
    
    "code_run": "There is a function called {func} in the following Python code.\n\n{context}\n\nPlease compute the exact value of {func_call}. The value of {func_call} is",
    
    "code_debug": "Following is a Python code where exactly one of the functions/methods has a deliberate error that makes it crash.\n\n{context}\n\nOptions:\nA. {OPTION_A}\nB. {OPTION_B}\nC. {OPTION_C}\nD. {OPTION_D}\n\nThe correct option is:",
    
    "longdialogue_qa_eng": "Below is a dialogue script where one random occurrence of a character name is replaced with \"$$MASK$$\", and you should try to guess who that character is.\n\n{context}\n\n{input} Just give the name without other words. Do not give me random numbers or something else. The name that has been replaced with \"$$MASK$$\" is ",
}

def infini_bench_create_prompt(eg: dict, data_name: str, template: str) -> str:
    # Code tasks
    if data_name == "code_run":
        find_result = re.findall(r"func_[0-9]+\(\-?[0-9]+\)", eg['input'])
        func_call = find_result[0]
        func = func_call.split("(")[0]
        return template.format(
            func=func,
            func_call=func_call,
            context=eg["context"],
        )
    elif data_name in ["code_debug", "code_debug_qa"]:
        code = eg["context"]
        if data_name == "code_debug":
            return template.format(
                context=code,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        return template.format(
            context=code,
        )
    # Long book tasks
    elif data_name in [
        "longbook_choice_eng",
        "longbook_qa_eng",
        "longbook_sum_eng",
        "longbook_qa_chn",
    ]:
        book = eg["context"]
        if data_name == "longbook_choice_eng":
            return template.format(
                question=eg["input"],
                context=book,
                OPTION_A=eg["options"][0],
                OPTION_B=eg["options"][1],
                OPTION_C=eg["options"][2],
                OPTION_D=eg["options"][3],
            )
        elif data_name == "longbook_qa_eng":
            return template.format(
                question=eg["input"],
                context=book,
            )
        elif data_name == "longbook_sum_eng":
            return template.format(
                context=book,
            )
        elif data_name == "longbook_qa_chn":
            return template.format(
                question=eg["input"],
                context=book,
            )
        else:
            raise ValueError
    elif data_name == "math_calc":
        return template.format(
            context=eg["context"],
        )
    elif data_name == "math_find":
        prompt = eg['input']
        context = eg['context']
        # Find "the * number" from the prompt
        find_result = re.findall(r"The .+ of", prompt)
        assert find_result, f"Cannot find the target number in {prompt}"
        target_number = find_result[0].lower()[:-3]
        # Replace the number with the answer
        prefix = f"What is {target_number} in the following list?"
        return template.format(
            prefix=prefix,
            context=context,
            input=prompt,
        )

    if "content" in eg:
        content = eg["content"]
        del eg["content"]
        eg["context"] = content

    format_dict = {
        "context": eg["context"],
        "input": eg["input"],
    }

    if data_name == "kv_retrieval":
        format_dict["input"] = eg["input"].split('"')[1]

    prompt = template.format(**format_dict)
    return prompt

def infini_bench_get_answer(eg: dict, data_name: str):
    if data_name in ["code_debug", "longbook_choice_eng"]:
        OPTIONS = "ABCD"
        if isinstance(eg["answer"], str):
            ret = [eg["answer"], OPTIONS[eg['options'].index(eg["answer"])]]
        elif isinstance(eg["answer"], list):
            if len(eg["answer"]) == 1:
                ret = [eg["answer"][0], OPTIONS[eg['options'].index(eg["answer"][0])]]
            elif len(eg["answer"]) == 2 and eg["answer"][1] in ['A', 'B', 'C', 'D']:
                ret = eg['answer']
            else:
                raise ValueError
        else:
            raise ValueError
        return ret

    return eg["answer"]