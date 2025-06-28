Question 1: The motivation for transitioning from FedReFT to FedReFT+ is not clearly articulated, making it difficult to understand the limitations of FedReFT that FedReFT+ aims to overcome. As described, FedReFT resembles a standard FedAvg approach where clients apply ReFT locally and the server aggregates the trainable parameters via weighted averaging, yet the paper does not clarify why this baseline is insufficient.

Answer:
We appreciate the reviewer’s thoughtful feedback. We would like to clarify the concerns raised by the reviewer:  There is no prior method called “FedReFT” upon which our work builds. Our proposed FedReFT+ is not an incremental improvement over an existing "FedReFT" method, but rather a new framework that introduces Representation Fine-Tuning (ReFT) into the federated learning (FL) setting for the first time, to the best of our knowledge. 
The name “FedReFT+” reflects our broader contributions beyond simply adapting ReFT to FL. Specifically: 
We identify that naively aggregating representation-level updates across heterogeneous clients can lead to semantic misalignment, especially in task-diverse FL settings. 
To address this, we propose a novel All-But-Me (ABM) aggregation strategy that improves semantic stability by letting clients partially incorporate global representation shifts without diluting their local semantics. We perform vanilla FedAvg of ReFT in the Federated Learning setting, which we could refer to as FedReFT, as shown in Appendix F.1, Figure 3. 

Question 2: The adoption of the geometric median in the All-but-Me aggregation strategy lacks strong justification. While an ablation study is provided in Appendix F, the performance differences between aggregation methods are marginal, and the geometric median introduces higher computational complexity compared to simpler alternatives like Mean-ABM.

Answer:
While the performance gap may appear small at first glance, we would like to emphasize that even modest improvements are meaningful in the FL setting, particularly when evaluating under highly heterogeneous task distributions. As shown in Figure 3 (Appendix F):
  - Commonsense task: Geometric Median-ABM improves over Mean-ABM by 0.41 points and over FedAvg by 0.73 points. 
  - Arithmetic reasoning (GSM8K): Geometric Median-ABM improves over Mean-ABM by 1.01 and over FedAvg by 1.86. 
  - GLUE (NLU task): The improvement over Mean-ABM is 0.19, and over FedAvg is 1.06. 

These improvements are consistent across three diverse task types. We chose Geometric Median-ABM not for peak accuracy alone but for its robustness to outlier updates and task drift, which are common in federated personalization scenarios. Especially in the arithmetic reasoning task, which is highly sensitive to client diversity, the gains were more pronounced. While we acknowledge the additional cost of computing the geometric median, this is done only once per round at the server on low-dimensional sparse intervention parameters (not full models), making the overhead negligible. Importantly, client-side efficiency is unaffected, preserving our design goal of being lightweight for edge devices. We will revise Appendix F to include variance/error bars and improve clarity in the camera-ready version. 

Question 3: 
The proposed FedReFT+ does not consistently outperform existing methods across all benchmarks. In both Table 3 and Table 5, several baselines achieve better performance, raising concerns about the practical utility and competitiveness of FedReFT+ in real-world applications despite the authors’ efforts to improve performance.  

Answer: The centralized standalone ReFT baseline does not consistently outperform other PEFT methods in accuracy; however, it is 15–65× more parameter-efficient than LoRA while still achieving competitive accuracy. Therefore, FedReFT+ delivers the best balance of Parameter efficiency and performance. We acknowledge that FedReFT+ may not achieve the highest score on every benchmark. However, when considering accuracy and parameter efficiency, FedReFT+ nearly outperforms all state-of-the-art PEFT methods in Federated Learning settings. The following tables shows show how much FedReFT+ efficient compre to the SOTA appraoches. 

Commonsense Reasoning Task using LlaMa-3.2 3B from Table 3:
| Method     | Rank | Param (M) | Eff. vs FedReFT+(r8) | Acc Δ vs FedReFT+(r8)| Eff. vs FedReFT+(r8) | Acc Δ FedReFT+(r8) |
|------------|------|-----------|----------------|------------------|----------------|------------------|
| FLoRA      | 32   | 243.15    | 22.09×         | –2.61%           | 88.42×         | –3.17%           |
| FedIT      | 32   | 48.63     | 4.42×          | +0.48%           | 17.68×         | –0.08%           |
| FFA-LoRA   | 32   | 24.31     | 2.21×          | +5.11%           | 8.84×          | +4.55%           |
| Fed-SB     | 120  | 2.73      | 0.25×          | +0.56%           | 0.99×          | 0.00%            |
| FedReFT+ | 32   | 11.01     | 1.00×          | 0.00%            | 4.00×          | +0.56%           |
| FedReF+ | 8    | 2.75      | 0.25×          | –0.56%           | 1.00×          | 0.00%            |


GLUE Tasks using RoBERTa-large model from Table 5:
| Method         | TP (M) | Param (%) | Avg Accuracy (%) | Efficiency vs FedReFT+ | Accuracy Δ vs FedReFT+ |
|----------------|--------|------------|-------------------|----------------------|----------------------|
| FFA-LoRA       | 1.44   | 0.405      | 89.39             | 27.17×               | +1.54%               |
| FedDPA-LoRA    | 2.62   | 0.737      | 89.47             | 49.43×               | +1.46%               |
| FedSA-LoRA     | 1.83   | 0.551      | 90.43             | 34.53×               | +0.50%               |
| FedReFT+ (ours)      | 0.053  | 0.015      | 90.93             |                |                 |


Arithmetic Reasoning Task using LLaMa-3 8B from Table 6:
| Method        | Rank | Param (M) | Param (%) | Accuracy (%) | Efficiency vs FedReFT+ | Accuracy Δ vs FedReFT+ |
|---------------|------|-----------|------------|----------------|----------------------|----------------------|
| FedSA-LoRA    | 8    | 30.40     | 0.38       | 46.63          | 7.25×                | +3.05%               |
| FFA-LoRA      | 8    | 15.20     | 0.19       | 46.32          | 3.63×                | +3.36%               |
| FedReFT+ (ours)    | 8    | 4.19      | 0.0622     | 49.68          |                 |                |

The tie-ϕ variant further demonstrates how our method scales down to even more compact configurations with acceptable performance loss. 
