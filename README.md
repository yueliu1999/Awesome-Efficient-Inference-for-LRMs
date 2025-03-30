# Awesome-Efficient-Inference-for-LRMs

Awesome-Efficient-Inference-for-LRMs is a collection of state-of-the-art, novel, exciting token efficient methods for Large Reasoning Models (LRMs). It contains papers, codes, datasets, evaluations, and analyses. Any additional things regarding efficient inference for LRMs, PRs, issues are welcome and we are glad to add you to the contributor list [here](#contributors). Any problems, please contact yliu@u.nus.edu. If you find this repository useful to your research or work, it is really appreciated to star this repository and cite our papers [here](#Reference). :sparkles:

<img src="./overview.png" alt="overview" style="zoom:61%;" />


## Bookmarks

- [Explicit Compact CoT](#explicit-compact-cot)
- [Implicit Latent CoT](#implicit-latent-cot)



## Papers



### Explicit Compact CoT

| Time    | Title                                                        | Venue |                  Paper                   |                             Code                             |
| ------- | ------------------------------------------------------------ | :---: | :--------------------------------------: | :----------------------------------------------------------: |
| 2025.03 | **Sketch-of-thought: Efficient llm reasoning with adaptive cognitive-inspired sketching (SoT)** |   arXiv'25     | [link](https://arxiv.org/abs/2503.05179) |        [link](https://github.com/SimonAytes/SoT)      |
| 2024.07 | **Concise Thoughts: Impact of Output Length on LLM Reasoning and Cost** |   arXiv'24     | [link](https://export.arxiv.org/abs/2407.19825) |        /      |
| 2025.02 | **Chain of Draft: Thinking Faster by Writing Less** |   arXiv'25     | [link](https://arxiv.org/abs/2502.18600) |        [link](https://github.com/sileix/chain-of-draft)      |
| 2024.12 | **Token-Budget-Aware LLM Reasoning** |   arXiv'24     | [link](https://arxiv.org/abs/2412.18547) |        [link](https://github.com/GeniusHTX/TALE)      |
| 2025.02 | **Meta-Reasoner: Dynamic Guidance for Optimized Inference-time Reasoning in Large Language Models** |   arXiv'25     | [link](https://arxiv.org/abs/2502.19918) |        /      |
| 2025.03 | **SOLAR: Scalable Optimization of Large-scale Architecture for Reasoning** |   arXiv'25     | [link](https://arxiv.org/abs/2503.04530) |        /      |
| 2024.12 | **C3oT: Generating Shorter Chain-of-Thought without Compromising Effectiveness** |   arXiv'24     | [link](https://arxiv.org/abs/2412.11664) |        /      |
| 2025.02 | **TokenSkip: Controllable Chain-of-Thought Compression in LLMs** |   arXiv'25     | [link](https://arxiv.org/abs/2502.12067) |        [link](https://github.com/hemingkx/TokenSkip)      |
| 2025.03 | **InftyThink: Breaking the Length Limits of Long-Context Reasoning in Large Language Models** |   arXiv'25     | [link](https://arxiv.org/abs/2503.06692) |        /      |
| 2025.02 | **LightThinker: Thinking Step-by-Step Compression** |   arXiv'25     | [link](https://arxiv.org/abs/2502.155892) |        [link](https://arxiv.org/abs/2412.21006)      |
| 2025.02 | **CoT-Valve: Length-Compressible Chain-of-Thought Tuning** |   arXiv'25     | [link](https://arxiv.org/abs/2502.09601) |        /      |
| 2024.07 | **Distilling System 2 into System 1** |   arXiv'24     | [link](https://arxiv.org/abs/2407.06023) |        /      |
| 2025.02 | **Self-Training Elicits Concise Reasoning in Large Language Models** |   arXiv'25     | [link](https://arxiv.org/abs/2502.20122) |        [link](https://github.com/TergelMunkhbat/concise-reasoning)      |
| 2024.11 | **Can Language Models Learn to Skip Steps?** |   arXiv'24     | [link](https://arxiv.org/abs/2411.01855) |        [link](https://github.com/tengxiaoliu/LM_skip)      |
| 2024.12 | **Verbosity-Aware Rationale Reduction: Effective Reduction of Redundant Rationale via Principled Criteria** |   arXiv'24     | [link](https://arxiv.org/abs/2412.21006) |        /      |
| 2025.02 | **DAST: Context-Aware Compression in LLMs via Dynamic Allocation of Soft Tokens** |   arXiv'25     | [link](https://arxiv.org/abs/2502.11493) |        /      |
| 2025.01 | **Kimi k1.5: Scaling Reinforcement Learning with LLMs** |   arXiv'25     | [link](https://arxiv.org/abs/2501.12599) |        /      |
| 2025.01 | **O1-Pruner: Length-Harmonizing Fine-Tuning for O1-Like Reasoning Pruning** |   arXiv'25     | [link](https://arxiv.org/abs/2501.12570) |        [link](https://github.com/StarDewXXX/O1-Pruner)      |
| 2025.03 | **Optimizing Test-Time Compute via Meta Reinforcement Fine-Tuning** |   arXiv'25     | [link](https://arxiv.org/abs/2503.07572) |        [link](https://cohenqu.github.io/mrt.github.io/)      |
| 2025.02 | **Training Language Models to Reason Efficiently** |   arXiv'25     | [link](https://arxiv.org/abs/2502.04463) |        [link](https://github.com/Zanette-Labs/efficient-reasoning)      |
| 2025.02 | **Anthropic. Claude 3.7 sonnet and claude code** |   /     | / |        [link](https://www.anthropic.com/news/claude-3-7-sonnet)      |
| 2025.03 | **L1: Controlling How Long A Reasoning Model Thinks With Reinforcement Learning** |   arXiv'25     | [link](https://arxiv.org/abs/2503.04697) |        [link](https://cmu-l3.github.io/l1/)      |
| 2025.02 | **Stepwise Perplexity-Guided Refinement for Efficient Chain-of-Thought Reasoning in Large Language Models** |   arXiv'25     | [link](https://arxiv.org/abs/2502.13260) |        /      |
| 2025.01 | **Think Smarter not Harder: Adaptive Reasoning with Inference Aware Optimization** |   arXiv'25     | [link](https://arxiv.org/abs/2501.17974) |        /      |



<img src="./takeaway1.png" alt="takeaway1" style="zoom:61%;" />



### Implicit Latent CoT



| Time | Title   |  Venue   | Paper | Code |
| ---- | ------- | :------: | :---: | :--: |
| 2023.11 | **Implicit Chain of Thought Reasoning via Knowledge Distillation** |   arXiv'23     | [link](https://arxiv.org/abs/2311.01460) |        [link](https://github.com/da03/implicit_chain_of_thought/)      |
| 2025.02 | **CODI: Compressing Chain-of-Thought into Continuous Space via Self-Distillation** |   arXiv'25     | [link](https://arxiv.org/abs/2502.21074) |        /      |
| 2024.05 | **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step** |   arXiv'24     | [link](https://arxiv.org/abs/2405.14838) |        [link](https://github.com/da03/Internalize_CoT_Step_by_Step)      |
| 2024.12 | **Training Large Language Models to Reason in a Continuous Latent Space** |   arXiv'24     | [link](https://arxiv.org/abs/2412.06769) |        /      |
| 2024.12 | **Compressed Chain of Thought: Efficient Reasoning Through Dense Representations** |   arXiv'24     | [link](https://arxiv.org/abs/2412.13171) |        /      |
| 2025.01 | **Efficient Reasoning with Hidden Thinking** |   arXiv'25     | [link](https://arxiv.org/abs/2501.19201v1) |        [link](https://github.com/shawnricecake/Heima)      |
| 2025.02 | **Token Assorted: Mixing Latent and Text Tokens for Improved Language Model Reasoning** |   arXiv'25     | [link](https://arxiv.org/abs/2502.03275) |        /      |
| 2025.02 | **SoftCoT: Soft Chain-of-Thought for Efficient Reasoning with LLMs** |   arXiv'25     | [link](https://arxiv.org/abs/2502.12134) |        /      |


<img src="./takeaway2.png" alt="takeaway1" style="zoom:61%;" />







## Reference

If you find this repository helpful to your research, it is really appreciated to cite our papers. :sparkles:

```

```



## Contributors

<a href="https://github.com/yueliu1999" target="_blank"><img src="https://avatars.githubusercontent.com/u/41297969?s=64&v=4" alt="yueliu1999" width="96" height="96"/></a> 
<a href="https://github.com/jiayingwu19" target="_blank"><img src="https://avatars.githubusercontent.com/u/54256470?v=4" alt="jiayingwu19" width="96" height="96"/></a> 
<a href="https://github.com/yf-he" target="_blank"><img src="https://avatars.githubusercontent.com/u/64448231?v=4" alt="yf-he" width="96" height="96"/></a> 
<a href="https://github.com/bhooi" target="_blank"><img src="https://avatars.githubusercontent.com/u/733939?v=4" alt="bhooi" width="96" height="96"/></a> 
<a href="https://github.com/Rohan-GRH" target="_blank"><img src="https://avatars.githubusercontent.com/u/166485256?v=4" alt="Rohan-GRH" width="96" height="96"/></a> 




<p align="right">(<a href="#top">back to top</a>)</p>












