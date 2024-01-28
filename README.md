# graduate_student_LLM
## Initiatives

Hello there!

I'm currently a Master of Science in Computer Science (MSCS) student at Yale University, focusing my research on Large Language Models (LLMs). In the world of graduate studies, particularly in fields like ours, conducting expansive and resource-intensive experiments – such as training a model from the scratch – can be quite challenging due to resource constraints.

With this in mind, my project aims to explore areas like prompt tuning, instruction tuning, Parameter-Efficient Fine-Tuning (PEFT), Mixture of Experts (MoE), and Retrieval-Augmented Generation (RAG), among others. One interesting observation is that these methods often share similarities in their implementation pipelines and evaluation metrics. Additionally, the variety of open-source models readily available for these kinds of tasks is somewhat limited.

The goal of this project is to develop a comprehensive pipeline that facilitates smaller-scale experiments and enhances understanding of LLMs. This will include thorough documentation and well-structured, objective-oriented code modules, designed to be accessible and useful for fellow researchers and enthusiasts in the field.

## Preliminary Plan
### Model
1. (Phase 1) Load from PTM - Llama
2. (Phase 1) Model Summary
3. (Phase 1) Generation / Log
4. (Phase 3) Support more models, e.g., Mistral, Phi, Alpaca etc.

### Data Processing
1. (Phase 1) Prefix-prompt
2. (Phase 1) K-shot
3. (Phase Unk) ?self-aligned prompting?
4. (Phase 1) Diverse input format: json(l) / txt / csv
5. (Phase 1) Data backup 
6. (Phase N) Instruction build-up -> Are u looking for promptsource?

### SFT / Model Revison
1. (Phase 2) SFT - QLora, IA3, Adapters, etc.
2. (Phase 2) MoE Layers - gating, expert model construction.
3. (Phase 2) Flexible metrics
4. (Phase 3) Load from frequent-used Eval/Benchmark

### Speed up
1. (Phase 3) FlashAttention
2. (Phase 3) Optimize Data I/O
3. (Phase Unk) DeepSpeed?
 

# graduate_student_LLM
## Initiatives
Hello there!

I'm currently a Master of Science in Computer Science (MSCS) student at Yale University, focusing my research on Large Language Models (LLMs). As a graduate student, particularly in fields like ours, conducting expansive and resource-intensive experiments – such as training a model from the ground up – can be quite challenging due to resource constraints.

With this in mind, my project aims to build up a flexible code base in areas like prompt tuning, instruction tuning, Parameter-Efficient Fine-Tuning (PEFT), Mixture of Experts (MoE), and Retrieval-Augmented Generation (RAG), among others. One interesting observation is that these methods often share similarities in their implementation pipelines and evaluation metrics. Additionally, the variety of open-source models readily available for these kinds of tasks is somewhat limited. Hope this would help all of us make it through.

The goal of this project is to develop a comprehensive pipeline that facilitates smaller-scale experiments and enhances understanding of LLMs for beginners. This will include thorough documentation and well-structured, objective-oriented code modules, designed to be accessible and useful for fellow researchers and enthusiasts in the field.

## Preliminary Plan
### Model
1. (Phase 1) Load from PTM - Llama
2. (Phase 1) Model Summary
3. (Phase 1) Generation / Log
4. (Phase 3) Support more models, e.g., Mistral, Phi, Alpaca etc.

### Data Processing
1. (Phase 1) Prefix-prompt
2. (Phase 1) K-shot
3. (Phase Unk) ?self-aligned prompting?
4. (Phase 1) Diverse input format: json(l) / txt / csv
5. (Phase 1) Data backup 
6. (Phase N) Instruction build-up -> Are u looking for promptsource?

### SFT / Model Revison
1. (Phase 2) SFT - QLora, IA3, Adapters, etc.
2. (Phase 2) MoE Layers - gating, expert model construction.
3. (Phase 2) Flexible metrics
4. (Phase 3) Load from frequent-used Eval/Benchmark

### Speed up
1. (Phase 3) FlashAttention
2. (Phase 3) Optimize Data I/O
3. (Phase Unk) DeepSpeed?
