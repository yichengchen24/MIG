# MIG

Welcome to MIG (**M**aximize the **I**nformation **G**ain) Project!

We will continue to update, please stay tuned!

## What is MIG?
MIG is an automatic data selection method for instruction tuning. It proposes an information-based dataset measurement that comprehensively evaluates data quality and diversity.

## News
* 📄 [04/2025] MIG paper [MIG: Automatic Data Selection for Instruction Tuning by Maximizing Information Gain in Semantic Space]() is released!

## Performance

### 🔦 Highlights
<img src="./assets/teaser.png" alt="x" width="400">
\
Comparison with different data selection methods:
* Sample 50K from the Tulu3 pool(939K).
* Training on Llama3.1-8B.
* Comprehensive evaluations including human-preference and knowledge-based benchmarks.

### 📈 Full Results


## 🏃‍♂️ How to start?
### Installation
* Create an environment
```shell
conda create -n xsample python=3.10
conda activate xsample
```
* Install pytorch (>2.0)
```shell
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia
```
* Install
```shell
git clone https://github.com/yichengchen24/xsample.git
cd xsample
pip install -e .
```
### Data Sampling


### SFT Training


## 💪 What's more?

We will continue to update including:

- [ ] Open-source sampled data and model weights
- [ ] Open-source processed data pool
- [ ] More automatic data selection strategies


## Citation
If you find the content of this project helpful, please cite our paper as follows:
