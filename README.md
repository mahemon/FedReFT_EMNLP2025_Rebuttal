## Reviewer 1: 
Weakness Question 1: The motivation for transitioning from FedReFT to FedReFT+ is not clearly articulated... 

Answer:
There is no prior method called “FedReFT” upon which our work builds, rather a new framework that introduces Representation Fine-Tuning (ReFT) into the federated learning (FL) setting for the first time, to the best of our knowledge. The name “FedReFT+” reflects our broader contributions beyond simply adapting ReFT to FL. Specifically, we identify that naively aggregating representation-level updates across heterogeneous clients can cause semantic misalignment, especially in task-diverse FL. To address this, we propose a novel All-But-Me (ABM) aggregation that improves semantic stability by letting clients partially incorporate global shifts without diluting local semantics.

Question 2: The adoption of the geometric median in the ABM aggregation strategy lacks strong justification...

Answer:
We would like to emphasize that even modest improvements are meaningful in the FL setting, particularly when evaluating under highly heterogeneous task distributions. As shown in Figure 3 and Table 1: The arithmetic mean has time complexity O(d), while the geometric median (Weiszfeld’s algo.) has O(T·d), with T as iterations and d as parameters. Both have memory complexity O(d). Despite being computationally more expensive, the geometric median is more robust for heterogeneous aggregation and often yields higher accuracy in FL settings in the following table:

|Task|Method|Accu (%)|Accu Δ (GeoMed vs. others)|Params (M)|
|-------------|-----------------|--------------|----------------------------|-----------------------|
|Commonsense,LLaMa-2 7B|FedAvg|70.26|+0.73%|4.70|
| |Mean_ABM|70.58|+0.41%|4.70|
| |GeoMedian_ABM|70.99|–|4.70|
||||||
|Arithmetic,LLaMa-2 7B|FedAvg|15.35|+1.86%|4.70 |
| |Mean_ABM | 16.20 | +1.01% | 4.70|
||GeoMedian_ABM|17.21|–|4.70|
||||||
|GLUE,RoBERTa|FedAvg|51.30 |+1.06% | 0.053|
| |Mean_ABM|52.17| +0.19%|0.053|
| |GeoMedian_ABM| 52.36|–|0.053|

These improvements hold across three diverse tasks. We chose Geometric Median-ABM for its robustness to outliers and task drift, not just peak accuracy. In arithmetic reasoning, gains were more pronounced due to high client diversity. Though geometric median adds server-side cost, it’s computed once per round on low-dimensional sparse parameters, keeping overhead negligible. Client-side remains lightweight. Appendix F will be revised for clarity and variance/error bars. 

Question 3:
The proposed FedReFT+ does not consistently outperform existing methods across all benchmarks...

Answer: The centralized ReFT baseline (Wu et al., 2024b) is 15–65× more parameter-efficient than LoRA while maintaining competitive accuracy, though it doesn’t consistently outperform other PEFT methods. Therefore, FedReFT+ delivers the best balance of Parameter efficiency and performance. While FedReFT+ may not achieve the highest score on every benchmark, it nearly outperforms all state-of-the-art PEFT methods in Federated Learning when considering both accuracy and parameter efficiency. From Table 3, FedReFT+ performance of LlaMa-3.2 3B across commonsense reasoning tasks with Mixed Task (MT) experimental setup:
|Method|Rank (R)|Param (M)|Avg Accu (%)|FedReFT+(R 32) Param Effi.|Accu Δ (FedReFT+(R 32) vs. others)|FedReFT+(R 8) Param Effi.|Accu Δ (FedReFT+(R 8) vs. others)|
|------------|----------|-----------|----------------|----------------------|---------------|---------------------|---------------|
|FLoRA|32|243.15|78.83|22.09×|–2.61%|88.42×|–3.17%|
|FedIT|32|48.63|75.74|4.42×|+0.48%|17.68×|–0.08%|
|FFA-LoRA| 32|24.31|71.11| 2.21×| +5.11%| 8.84×|+4.55%|
|Fed-SB|120|2.83| 75.66| 0.26×|+0.56%| 1.03×| 0.00%|
|FedReFT+|32|11.01|76.22|—|—| 4.00×| +0.56%|
|FedReFT+|8|2.75|75.66|0.25×|–0.56%|—|—|

From Table 5, which compares performance across GLUE tasks using the RoBERTa model, FedReFT+ employs rank 1 while all baselines use rank 8, and it outperforms all baselines with 27.17× to 34.53× higher parameter efficiency.
| Method|Trainable Param (M)|Avg Accu (%)|FedReFT+Param Effi.|Accu Δ (FedReFT+ vs. others)|
|---------------|--------|------------------|---------------------|---------------------|
|FFA-LoRA|1.44|89.39|27.17×|+1.54%|
|FedDPA-LoRA|2.62|89.47|49.43×|+1.46%|
|FedSA-LoRA|1.83|90.43|34.53×|+0.50%|
|FedReFT+|0.053|90.93|-|-|

From Table 6, Performance comparison on arithmetic reasoning tasks for GSM8K on LLaMa-3 8B model with LoRA
rank 8, which achieves (+3.05%, +3.36%) higher accuracy with (3.63×, 7.25×) times parameter efficient. 
|Method|Rank|Trainable Param (M)|Accu (%)|FedReFT+ Param Effi.|Accu Δ (FedReFT+ vs. others)|
|----------|------|---------|----------|---------|------------|
|FedSA-LoRA|8| 30.40|46.63|7.25×|+3.05%|
|FFA-LoRA|8|15.20|46.32|3.63×| +3.36%|
|FedReFT+|8|4.19|49.68|||

The tie-ϕ variant further demonstrates how our method scales down to even more compact configurations with acceptable performance loss. 

## Reviewer 2
Question 1: The authors claim that FedAvg can cause semantic interference ..

Answer:
ReFT methods operate on a frozen base model and learn task-specific Interventions on hidden representations. Our claim regarding semantic interference or collapse under FedAvg is grounded in the intuitive mismatch between LoReFT-style updates (ReFT, LoRA, LoReFT) and naive intervention parameter averaging, especially under heterogeneous client tasks. Since LoReFT modifies internal representation layers, averaging such updates across clients with divergent tasks can result in semantic drift, where the aggregated representation no longer aligns with any client’s local task semantics. As shown in Figure 3 and Table 1: 
|Task|Method|Accu (%)|Accu Δ (GeoMed vs. others)|Params (M)|
|-----------|------------|------------|---------|------------|
|Commonsense,LLaMa-2 7B|FedAvg|70.26|+0.73%|4.70|
||Mean_ABM|70.58|+0.41%|4.70|
||GeoMedian_ABM|70.99|–|4.70|
||||||
|Arithmetic,LLaMa-2 7B|FedAvg|15.35|+1.86%|4.70|
||Mean_ABM|16.20|+1.01%|4.70|
||GeoMedian_ABM|17.21|–|4.70|
||||||
|GLUE,RoBERTa|FedAvg|51.30|+1.06%|0.053|
||Mean_ABM|52.17|+0.19%|0.053|
||GeoMedian_ABM|52.36|–|0.053|

Question 2:
The authors claim that FedAvg can cause semantic interference...

Answer:
FedAvg on LoReFT is not well-established in the existing literature. We first tried FedAvg with LoReFT in FL (FedAvg in Figure 3 is actually FedAvg+LoReFT). Since it is a representation fine-tuning method, we aimed to preserve personalized fine-tuning locally at each client, rather than simply averaging on the server side. Our motivation arises from the hypothesis—grounded in empirical evidence and architectural intuition—that naive aggregation of representation-level interventions (via FedAvg) can lead to semantic drift or collapse, particularly in heterogeneous tasks. Regarding baseline selection: FLoRA, FedIT, FFA-LoRA, and FedSB are the closest state-of-the-art PEFT methods, using LoRA-like decompositions. We selected them intentionally to benchmark our method against the strongest available alternatives in low-rank or PEFT-based FL.

Question 3:
The W2 could be partly answered by the contents from Appendix F. ... 
Answer:
We did the experiments on a reduced dataset for this section, so the results are not the same as the main context. We present the same results in Figure 3 and Table 1, which shows the result of the GLUE task on ROBERTa, Commonsense, and Arithmetic reasoning task on the LLaMA-2 7B model for three clients. The table in Answer 1 can also be considered for this question. 

Question 4:
The heterogeneous distribution among clients may not always occur...
Answer:
We conducted commonsense reasoning experiments in the Mixed Task (MT) setup, but due to space constraints, the results were moved to Appendix G.1 (Table 14). While the caption of Table 14 does not explicitly mention the commonsense reasoning task, we clarify that the experiments presented there are indeed conducted on the commonsense reasoning dataset. 

- There is no description about DT/MT for 3.3 Natural Language Understanding

Answer: For the NLU task experiments do not follow the DT/MT task distribution paradigm. Instead, we partitioned each GLUE task among clients, allowing local training on task-specific subsets. We applied this experimental design uniformly across all GLUE datasets. The primary objective was to examine whether lightweight intervention tuning can effectively align representations across clients within a single NLU task. 

- Table 4 gives DT results for 3.2 Arithmetic Reasoning; however ..

Answer: In the Mixed Task (MT) setting, each client trains on a subset of a combined reasoning dataset, encouraging generalization across tasks. Conversely, in the Distinct Task (DT) setting, each client trains on a unique reasoning task, enabling more personalized fine-tuning while benefiting from global updates. As expected, the DT setup typically results in higher accuracy, as clients specialize in a single task. 

- (2) No comparisons are made to other baselines in Table 4..

Answer: Regarding comparisons in Table 5, thre is no existing works follow the DT/MT experimental design over math10k task, which limited our ability to include direct comparisons in that table. However, we referenced several relevant works on arithmetic reasoning tasks on the GSM8K dataset and included comparative results in Table 6, which shows performance comparison on arithmetic reasoning tasks for GSM8K on the LLaMa-3 8B model with LoRA rank 8:
|Method|Rank|Trainable Param (M)|Accu (%)|FedReFT+ Param Effi.|Accu Δ (FedReFT+ vs. others)|
|----------|------|---------|----------|---------|--------|
|FedSA-LoRA|8| 30.40|46.63|7.25×|+3.05%|
|FFA-LoRA|8|15.20|46.32|3.63×| +3.36%|
|FedReFT+|8|4.19|49.68|||

Question 5:
The related work section is placed in the appendix ... 

Answer:
Due to page limitations, we moved the Related Work, ABM analysis, and ablation study to the appendix to prioritize core content in the main paper. If accepted, we will include concise versions of these important sections in the camera-ready version to improve clarity and completeness. 

Question 6:
In the author's checklist ... 

Answer:
We sincerely apologize for the oversight; our work uses only publicly available, licensed datasets that do not contain any B4 PII or offensive content. We will correct the mistaken selection of item B4 in the camera-ready version. 



## Reviewer 3
**Question 1:** In Table 3, FedReFT+ with rank=8 shows no improvement over Fed-SB in terms of both performance and the number of trainable parameters. 

**Answer:**
The following table shows the performance efficiency and accuracy status of FedReFT+ over other baselines. FedReFT+ with rank 8 achieves a similar accuracy of 75.66%, while being 1.03× more parameter-efficient compared to Fed-SB. In contrast, Fed-SB utilizes a much higher LoRA rank of 120, which significantly increases the number of trainable parameters and contributes to its comparable accuracy. From Table 3, Federated fine-tuning performance of LlaMa-3.2 3B across five commonsense reasoning tasks with Mixed Task (MT) experimental setup, where clients train on heterogeneous task mixtures to promote generalizable representations.
| Method     | Rank (R) | Param (M) | Avg Accu (%) | FedReFT+(R 32) Param Effi. | Accu Δ (FedReFT+(R 32) vs. others)| FedReFT+(R 8) Param Effi. | Accu Δ (FedReFT+(R 8) vs. others)|
|------------|----------|-----------|----------------|-----------------------|----------------|----------------------|----------------|
| FLoRA      | 32       | 243.15    | 78.83          | 22.09×                | –2.61%         | 88.42×               | –3.17%         |
| FedIT      | 32       | 48.63     | 75.74          | 4.42×                 | +0.48%         | 17.68×               | –0.08%         |
| FFA-LoRA   | 32       | 24.31     | 71.11          | 2.21×                 | +5.11%         | 8.84×                | +4.55%         |
| Fed-SB     | 120      | 2.83      | 75.66          | 0.26×                 | +0.56%         | 1.03×                | 0.00%          |
| FedReFT+| 32       | 11.01     | 76.22          | —                     | —              | 4.00×                | +0.56%         |
| FedReFT+| 8        | 2.75      | 75.66          | 0.25×                 | –0.56%         | —                    | —              |

**Question 2:** Table 4 displays the performance of FedReFT+ across different models with two setups. However, it is unclear what the baseline performance is under these settings.

**Answer:** We did not find any reference work on arithmetic reasoning in the Distinct Task and Mixed Task experiments setup. We found some baseline work on the arithmetic reasoning task over the GSM8K dataset, and we compare FedReFT+ in Table 6 for the GSM8K Dataset. FedReFT+ achieves 7.25× and 3.63× parameter efficiency, along with +3.05% and +3.36% accuracy improvements over FedSA-LoRA and FFA-LoRA, respectively.

**Question 3:** Table 5 only presents performance on four GLUE subtasks. A more comprehensive comparison across all subtasks would provide a clearer analysis. 

**Answer:** We have taken the performance results of all baseline methods from (Guo et al., 2024). Therefore, we did experiments on these six GLUE subtasks to compare with the well-established baselines. These results across six tasks demonstrate significant performance gains, achieving 27.17× to 49.43× parameter efficiency compared to the SOTA baselines.
| Method        | RANK | Param (M) | FedReFT+ Param Effi.| MNLI-m | MNLI-mm | SST-2 | QNLI | QQP  | RTE | Avg |
|---------------|----|-----|-----|--------|---------|-------|------|------|-------|--------|
| FFA-LoRA      | 8 | 1.44 | 27.17× | 88.83  | 88.27   | 94.95 | 91.52| 86.71| 86.08 | 89.39 |
| FedDPA-LoRA   | 8 | 2.62| 49.43× | 88.99  | 88.43   | 95.50 | 90.74| 85.73| 87.44 | 89.47 |
| FedSA-LoRA    | 8 | 1.83| 34.53×| 90.18  | 88.88   | 96.00 | 92.13| 87.48| 87.93 | 90.43 |
| FedReFT+ | 1 | 0.053| - |88.86  | 89.61   | 95.17 | 94.52| 86.57| 87.80 | **90.46** |

**Question 4:** Figure 1 is unclear, presenting both the count and percentage of trainable parameters. Since these two metrics are essentially the same, a performance comparison relative to the number of trainable parameters would be more intuitive. 

**Answer:** 
We redesigned Figure 1, which is a Illustration of the relationship between the average accuracy (in $\%$) and trainable parameter (in $\%$) for various federated PEFT methods on Commonsense, Arithmetic, and GLUE benchmarks using LLaMA-3.2B, LLaMA-3 8B, and RoBERTa-large models, respectively. The figure can be found [in this anonymous URL](https://postimg.cc/gw1PRqPR)

**Comments, Suggestions, And Typos:**
- Section 2.1 compares various intervention parameter sharing strategies, but FedReFT+ ultimately selects Full Intervention Sharing without any special design. Therefore, this discussion could be moved to the Appendix. In contrast, the ablation study on the Aggregation method is important and should be included in the main body of the paper. 

**Answer:** We will correct the typos in the camera-ready version. We are considering moving Section 2.1 to the Appendix to improve the flow of the main content. Additionally, we will incorporate the ablation study on the aggregation method into the main paper to highlight its significance. 

- Many tables and figures are not located on the same pages where they are referenced, which makes the paper hard to follow. 

**Answer:** We sincerely acknowledge this issue and will address it in the camera-ready version of the paper.

