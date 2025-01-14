# ShiftAddLLM

[![License: MIT](https://img.shields.io/badge/License-MIT-success.svg)](https://opensource.org/licenses/MIT)

This is PyTorch implementation of [ShiftAddLLM: Accelerating Pretrained LLMs via Post-Training Multiplication-Less Reparameterization](missing arxiv link)

Submitted to NeurIPS2024

## Table of Content
<!-- [TOC] -->
<div class="toc">
<ul>
<li><a href="#shiftaddllm">ShiftAddLLM</a><ul>
<li><a href="#table-of-content">Table of Content</a></li>
<li><a href="#introduction">Introduction</a></li>
</ul>
</li>
<li><a href="#basic-usage">Basic Usage</a><ul>
<li><a href="#setup">Setup</a></li>
<li><a href="#core-training-options">Core Training Options</a></li>
</ul>
</li>
<li><a href="#experiments">Experiments</a><ul>
<li><a href="#oursacc">Ours(Acc.)</a></li>
<li><a href="#ourslat">Ours(Lat.)</a></li>
<li><a href="#zeroshot">Zeroshot</a></li>
</ul>
</li>
<li><a href="#citation">Citation</a></li>
<li><a href="#acknowledgement">Acknowledgement</a></li>
</ul>
</li>
</ul>
</div>

## Introduction

-  [GPTQ](https://arxiv.org/abs/2210.17323) uses gradient-based weight quantization and develops INT3/4 kernels to reduce data movements, and [LUT-GEMM](https://arxiv.org/abs/2206.09557) further eliminates the dequantization and utilizes custom look-up table (LUT)-based CUDA kernels for reducing memory and computation costs.
- **Limitation**: However, these methods still rely on costly multiplication operations for FLOPs in both the attention and MLP layers of an LLM.
- **Our Contributions**:
    + Inspired by hardware practices, we propose accelerating pretrained LLMs via a post-training bitwise shift and add reparameterization towards efficient multiplication-less LLMs dubbed **ShiftAddLLM**. Specifically, all weights are quantized        into binary matrices paired with groupwise scaling factors; the associated multiplications are reparameterized into (1) shifts between activations and scaling factors and (2) queries and adds according to the binary matrices.
    + We propose a multi-objective optimization method to mitigate the accuracy drop associated with the shift and add reparameterization. This method aligns and optimizes both weight and output activation objectives, ensuring that the overall          reparameterization error for LLMs is minimized, thus achieving lower perplexity and better task accuracy.
    + We introduce a mixed and automated bit allocation strategy that automatically determines the optimal number of bits for reparameterized weights in each layer, based on their vulnerability to compression. Layers more susceptible to
      compression receive higher-bit representations, while less sensitive layers are assigned lower-bit representations.

## Basic Usage
### Setup
```
conda env create -f environment.yml
conda activate shiftaddllm
export PYTHONPATH='YOUR-PATH-TO-SHIFTADDLLM-REPO'
```

### Core Training Options
- `model`: huggingface path of the model to quantize.
- `dataset`: which dataset you want to use as calibration data.
- `wbits`: number of bits to use for quantization; use 16 for evaluating base model.
- `groupsize`: groupsize to use for quantization; default uses full row.
- `act-order`: whether to apply the activation order GPTQ heuristic.
- `bcq`: whether to quantize weights with binary coded quantization (bcq).
- `tcq`: whether to apply ternary coded quantization instead of bcq.
- `bcq_round`: steps to iterate bcq quantization.
- `columnwise`: whether to use columnwise - bcq - round to power of 2 - quantization to evaluate model.
- `block_quant` & `cust_group`: whether to use blockwise (8 column by 1/8 rows for 1 quantize param) - bcq - round to power of 2 - quantization to evaluate model. Need to use with 'columnwise' set.
- `use_bst`: whether to use binary search to get BinaryWeight.
- `apot_nums`: set nums shift weight for quantization.
- `acc`: whether to use Ours(acc.) to quantize the model.
- `lat`: whether to use Ours(lat.) to quantize the model. Only one of `acc` and `lat` should be set.

## Experiments
### Ours(Acc.)
To quantize LLMs using Ours(Acc.) method and evaluate the quantized LLMs performance, we provide scripts for five different LLMs family.
- [OPT](script/acc/eval_opt.sh)
- [Llama2 & Llama3](script/acc/eval_llama.sh)
- [Bloom](script/acc/eval_bloom.sh)
- [Mistral](script/acc/eval_mistral.sh)
- [Gemma](script/acc/eval_gemma.sh)

### Ours(Lat.)
To quantize LLMs using Ours(Lat.) method and evaluate the quantized LLMs performance, we provide scripts for five different LLMs family.
- [OPT](script/lat/eval_opt.sh)
- [Llama2 & Llama3](script/lat/eval_llama.sh)
- [Bloom](script/lat/eval_bloom.sh)
- [Mistral](script/lat/eval_mistral.sh)
- [Gemma](script/lat/eval_gemma.sh)

### Zeroshot
To evaluate quantized LLMs on seven downstream tasks for zero-shot task accuracy evaluation, run:
```
python3 main.py  <model_name> <calibration_dataset> --task <task_name> --num_fewshot <num_fewshot> 
```
 We also provide example scripts for two LLMs family.
- [OPT](zeroShot/script/eval_opt.sh)
- [Llama2 & Llama3](zeroShot/script/eval_llama.sh)
