################################################################################
# Copyright (c) 2024, NVIDIA CORPORATION. and affiliates
# All rights reserved.
#
# This source code is licensed under the Apache-2.0 license found in the
# LICENSE file in the root directory of this source tree.
################################################################################
# Modification Copyright 2024 ByteDance Ltd. and/or its affiliates.
################################################################################

# Model and Tokenizer
SEQ_LENGTHS=(
    65536
    #131072
)

MODEL_NAME=$1
MODEL_TEMPLATE_TYPE=$2

echo "Model Name: $MODEL_NAME"
echo "Model Template Type: $MODEL_TEMPLATE_TYPE"

# Benchmark and Tasks
NUM_SAMPLES=96
REMOVE_NEWLINE_TAB=false
STOP_WORDS=""

if [ -z "${STOP_WORDS}" ]; then
    STOP_WORDS=""
else
    STOP_WORDS="--stop_words \"${STOP_WORDS}\""
fi

if [ "${REMOVE_NEWLINE_TAB}" = false ]; then
    REMOVE_NEWLINE_TAB=""
else
    REMOVE_NEWLINE_TAB="--remove_newline_tab"
fi

# task name in `synthetic.yaml`
synthetic=(
    "niah_single_1"
    "niah_single_2"
    "niah_single_3"
    "niah_multikey_1"
    "niah_multikey_2"
    "niah_multivalue"
    "niah_multiquery"
    "niah_multiturn_1"
    "niah_multiturn_2"
    "vt"
    "fwe"
    "qa_1"
    "qa_2"
)

for MAX_SEQ_LENGTH in "${SEQ_LENGTHS[@]}"; do
    
    RESULTS_DIR="data/${MODEL_TEMPLATE_TYPE}/${MAX_SEQ_LENGTH}"
    DATA_DIR="${RESULTS_DIR}/"
    mkdir -p ${DATA_DIR}
    
    for TASK in "${synthetic[@]}"; do
        echo "TASK: ${TASK}, MAX_SEQ_LENGTH: ${MAX_SEQ_LENGTH}"
        python prepare.py \
            --save_dir ${DATA_DIR} \
            --task ${TASK} \
            --tokenizer_path ${MODEL_NAME} \
            --tokenizer_type hf \
            --max_seq_length ${MAX_SEQ_LENGTH} \
            --model_template_type ${MODEL_TEMPLATE_TYPE} \
            --num_samples ${NUM_SAMPLES} \
            ${REMOVE_NEWLINE_TAB}
    done

done