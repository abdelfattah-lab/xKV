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


Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\n",
    'lwm': "You are a helpful assistant.\nUSER: {ctx}\nASSISTANT: Answer: ",
    'qwen': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|system|>\nYou are a helpful assistant<|end|>\n<|user|>\n{ctx}<|end|>\n<|assistant|>\n",
}

Chat_Templates = {
    'base': "{msg}",
    'llama-3': "<|start_header_id|>user<|end_header_id|>{msg}<|eot_id|><|start_header_id|>assistant<|end_header_id|>",
    'yi': "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n",
    'glm': "<|user|>\n{msg}<|assistant|>\n",
    'lwm': "\nUSER: {msg}\nASSISTANT: ",
    'qwen': "<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n",
    'phi': "<|user|>\n{msg}<|end|>\n<|assistant|>\n",
}

Prefix_Templates = {
    'base': "{ctx}",
    'llama-3': "<|start_header_id|>system<|end_header_id|>You are a helpful assistant<|eot_id|><|start_header_id|>user<|end_header_id|>{ctx}<|eot_id|><|start_header_id|>assistant<|end_header_id|>OK! I will help you with that. Please ask me anything.<|eot_id|>",
    'yi': "<|im_start|>system\nYou are a helpful assistant<|im_end|>\n<|im_start|>user\n{ctx}<|im_end|>\n<|im_start|>assistant\nOK! I will help you with that. Please ask me anything.\n",
    'glm': "<|system|>\nYou are a helpful assistant\n<|user|> \n{ctx}<|assistant|>\nOK! I will help you with that. Please ask me anything.\n",
}

