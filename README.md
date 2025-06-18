<div align="center">
<h1>
  <img 
    src="static/images/xKV_logo_cute.png" 
    style="height: 33px; vertical-align: middle; margin-right: 0.3em; margin-bottom: -4px;"
    alt="xKV Logo"
  />
  xKV: Cross-Layer SVD for KV-Cache Compression
</h1>

Chi-Chih Chang<sup>1</sup>, 
Chien-Yu Lin<sup>2</sup>, 
Yash Akhauri<sup>1</sup>, 
Wei-Cheng Lin<sup>3</sup>,<br>
Kai-Chiang Wu<sup>3</sup>, 
Luis Ceze<sup>2</sup>, 
Mohamed S. Abdelfattah<sup>1</sup>


<sup>1</sup> Cornell University,   <sup>2</sup>University of Washington,<br><sup>3</sup>National Yang Ming Chiao Tung University<br>
[<a href="https://arxiv.org/abs/2503.18893">Paper</a>] | [<a href="https://abdelfattah-lab.github.io/xKV/">Website</a>]

</div>
<div align="center">
<img src="static/images/overview.jpg" align="top"/>
<figcaption>xKV Framework</figcaption>
</div>

## Updates
- [2025.03.24]:🚀 We release the 1st version of arXiv and code of xKV

## TL;DR
We introduce xKV, a simple yet effective post-training compression method for KV-Cache, leveraging inter-layer redundancy. By applying singular value decomposition (SVD) across group of layers, xKV achieves up to 8× compression of the KV-Cache while maintaining strong accuracy.

## Environment Setup
1. Clone the repository (Make sure you have Git, Conda installed on your system)
```
git clone https://github.com/abdelfattah-lab/xKV.git
cd xKV
```

2. Prepare environment
```
conda create -n xKV python=3.10
conda activate xKV

# cuda-toolkit (optional if your system doesn't have it)
conda install -y nvidia/label/cuda-12.4.0::cuda-toolkit
conda install -y nvidia::cuda-cudart-dev

# install dependency
pip install -r requirements.txt
pip install flash-attn==2.7.4.post1 --no-build-isolation
```

3. Create Datasets (for RULER evaluation only)
```
python -c "import nltk; nltk.download('punkt')"
cd evaluate/data/ruler
bash create_dataset.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" "llama-3"
```

## Accuracy Evaluations
We provide an evaluation script `evaluate/eval_acc.py` to measure the accuracy impact of compressing the KV-Cache with three different methods included in our paper:
1. Minicache
2. Single SVD
3. xKV

### Key Arguments
+ `--model_name_or_path`: Path or name of the model to evaluate (e.g., meta-llama/Meta-Llama-3.1-8B-Instruct).
+ `--xKV`: Toggle for enabling comprssion.
+ `--dataset_name` Comma-separated list of datasets (e.g., ruler/niah_single_1,ruler/niah_single_2,...).
+ `--layer_group_size`: Number of layers to be grouped.
+ `--rank_k`, `--rank_v`: Ranks used for each group of layers. 
+ `--layer_merge_impl` Target compression approaches [svd(default), slerp].
  

> [!NOTE] 
> When increasing the layer group size, you often need to adjust these ranks for a fair comparison. For instance, if you use `rank_k=128` for `layer_group_size=1`, then to compare performance under `layer_group_size=2`, set `rank_k=256` so that the average rank per layer is similar.
---

### Evaluation on RULER Benchmark
Below we provide the example commands for running the RULER benchmarks with different suppoted KV-Cache compression results.
#### xKV 
Enables xKV compression for all layers (start_layer_idx=0 to end_layer_idx=-1), grouping every 4 layers (layer_group_size=4), using ranks 512 and 768 for each grouped keys and values.
```
# xKV-4
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py --datalen 65536 --batch_size 1 --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --xKV --merge_k --merge_v --rank_k 512 --rank_v 768 --layer_group_size 4 --start_layer_idx 0 --end_layer_idx -1
```

#### Single SVD
For evaluation of Single SVD under similar compression level, replacing the arguments `--layer_group_size 1` and `--rank_k 128 --rank_v_192`.

```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py --datalen 65536 --batch_size 1 --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --xKV --merge_k --merge_v --rank_k 128 --rank_v 192 --layer_group_size 1 --start_layer_idx 0 --end_layer_idx -1
```

#### MiniCache
This command enables the MiniCache approach by specifying `--layer_merge_impl slerp`. The layers 16 through 31 are compressed.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py --datalen 65536 --batch_size 1 --dataset_name "ruler/niah_single_1,ruler/niah_single_2,ruler/niah_multikey_1,ruler/niah_multikey_2,ruler/niah_multiquery,ruler/niah_multivalue,ruler/vt,ruler/fwe,ruler/qa_1,ruler/qa_2" --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --xKV --merge_k --merge_v --layer_merge_impl slerp --layer_group_size 2 --start_layer_idx 16 --end_layer_idx 31
```

#### Customized Merge Config
We also support customized merge config by providing a yaml file to the `--customized_merge_config` argument. By writing a yaml file you can experiment with different merging groups and different ranks for each group. Please refer to the [configs/example.yaml](configs/example.yaml) for the format. 
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 evaluate/eval_acc.py --datalen 65536 --batch_size 1 --dataset_name "ruler/niah_single_1" --model_name_or_path meta-llama/Meta-Llama-3.1-8B-Instruct --xKV --customized_merge_config e
xample.yaml 
```


### Evalaution on DeepSeek Models
DeepSeek’s MLA (multi-latent attention) architecture has two types of hidden states that can be cached during inference:
+ Non-RoPE Latents (the learned, position-agnostic latent vectors).
+ RoPE-based Key States (rotary-positioned keys).
We reuse the Key and Value compression interfaces for these two elements:
+ `--merge_k` and `--rank_k` control compression of the non-RoPE latents (treated like “Keys”).
+ `--merge_v` and `--rank_v` control compression of the RoPE-based Key states (treated like “Values”).
In our paper, we focus on compressing only the non-RoPE latents only.

#### xKV for DeepSeek (compress only non-RoPE latents)
Enables xKV compression for all layers (start_layer_idx=0 to end_layer_idx=-1), grouping every 4 layers (layer_group_size=4), using ranks 512 for grouped latents.
```
CUDA_VISIBLE_DEVICES=0,1,2,3 OMP_NUM_THREADS=48 torchrun --standalone --nnodes=1 --nproc_per_node 4 \
evaluate/eval_acc.py \
--datalen 65536 \
--batch_size 1 \
--dataset_name "long_bench/repobench-p" \
--model_name_or_path deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct \
--xKV \
--merge_k \
--rank_k 512 \
--layer_group_size 4 \
--start_layer_idx 0 \
--end_layer_idx -1 \
--flash2
```

## Upcoming Roadmap
- [x] Accuracy Evaluation
- [ ] Release end-to-end system and efficiency evalution.
- [ ] Integration with sparse attention (e.g., ShadowKV)

## Citation
If you find xKV useful or relevant to your project and research, please kindly cite our paper:
```bibtex
@article{chang2025xkv,
  title = {xKV: Cross-Layer SVD for KV-Cache Compression},
  author = {Chang, Chi-Chih and Lin, Chien-Yu and Akhauri, Yash and Lin, Wei-Cheng and Wu, Kai-Chiang and Ceze, Luis and Abdelfattah, Mohamed S.},
  year = {2025},
  journal = {arXiv preprint arXiv:2503.18893},
  year = {2025}
}
```

## Acknowledgement
The evaluation scripts are built upon [ShadowKV](https://github.com/bytedance/ShadowKV) and [Palu](https://github.com/shadowpa0327/Palu) repository.