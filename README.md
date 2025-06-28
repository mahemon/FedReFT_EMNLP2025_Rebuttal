Reviewer 1: 
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




## Rivewer 2
Question 1: The authors claim that FedAvg can cause semantic interference or collapse for LoReFT in FL settings. However, the claim of collapsing is not supported by analysis or experiments. 

Answer: 
We appreciate the reviewer highlighting this point. Our claim regarding semantic interference or collapse under FedAvg is grounded in the intuitive mismatch between LoReFT-style updates (e.g., ReFT, LoRA, LoReFT) and naive parameter averaging, especially under heterogeneous client data or tasks. Since LoReFT modifies internal representation layers, averaging such updates across clients with divergent distributions can result in semantic drift, where the aggregated representation no longer aligns with any client’s local task semantics. 

Question 2: 
The authors claim that FedAvg can cause semantic interference or collapse for LoReFT in FL settings. However, the claim of collapsing is not supported by analysis or experiments. All the discussions about existing literature and challenges talk about the potential incompatibility between FedAvg and LoReFT. In the experiments, the comparisons are made against FLoRA, FedIT, FFA-LoRA, FedSB, etc. None of these methods is analyzed or discussed to show the methodologies used by these methods and how these methods would solve the aforementioned problems. Given the names of these methods, it could be that these methods are not related to LoReFT. In this case, the authors should discuss/show what happens if the FedAvg and LoReFT are directly combined and implemented. In its current form, the research questions brought up in the introduction are left unanswered. 

Answer:

 


Question 3: 
The W2 could be partly answered by the contents from Appendix F.1. However, Fig.3 in Appendix F.1 shows that the proposed method (ABM) provides marginal improvements to FedAvg. In addition, the results shown by Fig.3 in Appendix F.1 are not consistent with the results in the main context. No parameters about the used models are given for Appendix F.1. 
Answer:


Question 4:
The heterogeneous distribution among clients may not always occur, although it could appear in some real-world FL applications. The The Distinct Task (DT) scenario is closer to the heterogeneous distribution assumption. However, no DT results are given for 3.1 Commonsense Reasoning; only Mixed Task (MT) results are shown in Table 3. There is no description about DT/MT for 3.3 Natural Language Understanding. Table 4 gives DT results for 3.2 Arithmetic Reasoning; however, (1) models perform better in DT rather than MT; (2) no comparisons are made to other baselines. 

Answer: 
We conducted commonsense reasoning experiments under the Distinct Task (DT) and Mixed Task (MT) setups, as described in the paper. Due to space constraints, we moved the MT experiments to Appendix G.1 (Table 14). While the caption of Table 14 does not explicitly mention the commonsense reasoning task, we clarify that the experiments presented there are indeed conducted on the commonsense reasoning dataset. 

- There is no description about DT/MT for 3.3 Natural Language Understanding
Answer: For the Natural Language Understanding (NLU) experiments, we used the GLUE benchmark. However, these experiments do not follow the DT/MT task distribution paradigm. Instead, we partitioned each GLUE task among clients, allowing local training on task-specific subsets. Each client was evaluated using its local test set from the same task. We applied this experimental design uniformly across all GLUE datasets. Since this setup does not involve cross-task distribution (DT/MT), we did not include it under those sections. The primary objective was to examine whether lightweight intervention tuning can effectively align representations across clients within a single NLU task. 

- Table 4 gives DT results for 3.2 Arithmetic Reasoning; however, (1) models perform better in DT rather than MT; 
Answer: In the Mixed Task (MT) setting, each client trains on a subset of a combined reasoning dataset, encouraging generalization across tasks. Conversely, in the Distinct Task (DT) setting, each client trains on a unique reasoning task, enabling more personalized fine-tuning while benefiting from global updates. As expected, the DT setup typically results in higher accuracy, as clients specialize in a single task. 

- (2) no comparisons are made to other baselines in Table 4. 
Answer: Regarding comparisons in Table 5, to the best of our knowledge, no existing works follow the DT/MT experimental design over math10k, which limited our ability to include direct comparisons in that table. However, we referenced several relevant works on arithmetic reasoning tasks on the GSM8K dataset and included comparative results in Table 6. FedReFT+ achieved over 3+% accuracy gain compared to the SOTA. 

Question 5:
The related work section is placed in the appendix to bypass the page limit. Some content in the Appendices is important to the paper and should also be included in the main context, such as some portion of the related work, the analysis of ABM, and the ablation study. 

Answer: 
Thank you for pointing this out. Due to page limitations, we moved the Related Work, ABM analysis, and ablation study to the appendix to prioritize core content in the main paper. If accepted, we will include concise versions of these important sections in the camera-ready version to improve clarity and completeness. 

Question 6:
In the author's checklist, the authors have checked B4 PII and Offensive information, but no required elaboration/explanation is given in the annotated Section 3, nor in the whole paper. 

Answer: 
We sincerely apologize for the oversight; our work uses only publicly available, licensed datasets that do not contain any B4 PII or offensive content. We will correct the mistaken selection of item B4 in the camera-ready version. 

