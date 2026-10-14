# ES 673: NeuroAI

# Language Models

* **Due Date:** October 15th, 2026, 10 PM  
* **Deliverables:** Code Implementation (Python/PyTorch), Experiments, Methodological Explorations, and a Short Report. Kindly submit your code files, report (as pdf) on Classroom.  
* **AI Usage:** Generative AI models could be used for code implementation but kindly make sure you understand all the aspects of the code. The report should be written yourself. Marks would be deducted if found otherwise. 

### **Designing and Evaluating Cognitive and Language Models**

#### **1\. Design a Continuous Interaction Model**

Design a model that can interact continuously with a dynamic environment and learn object associations over time, basically going from a stationary model like LLMs to more real like human \-like learning technique. The model should not just passively receive data, but actively explore, associate, and adapt based on its interactions.

* Describe the model structure in terms of units, architecture  
* Specify how the model would learn.  
* Explain how the model handles continuous interaction.

There are no “right” or “wrong” answers. The goal is to **design a conceptually plausible system** inspired by principles we have discussed in class so far.

**2\. Ablation Study on LLAMA Model**

Perform an ablation study on an LLM (LLAMA, GPT, others) model to investigate which components are critical for performance and where the model breaks.

You can choose a set of ablation techniques (e.g., removing attention heads, zeroing out layers, masking input embeddings). Define simple downstream tasks (e.g., next-word prediction, sentence completion, sentiment classification). Measure performance drop or failure point as more components are ablated.

Please highlight tables or graphs showing the effect of ablation on model performance. Also discuss the insights gained (e.g., redundancy in architecture, sensitivity of semantic vs. syntactic processing).

**3\. Syntax vs. Semantics Analysis in Open-Source Language Models** 

Choose an open-source language model (e.g., LLAMA, GPT-2, GPT-J, BLOOM) and design an experiment to test how well it differentiates between syntactic structure and semantic meaning.

You can create or identify datasets that systematically vary syntax while keeping meaning constant, and vice versa (e.g., syntactically correct but semantically nonsensical sentences). Propose a methodology to evaluate the model’s sensitivity to syntax and semantics (e.g., perplexity scores, prediction accuracy, embedding similarity).

A brief report explaining your experimental design, expected outcomes, and interpretation of how syntax and semantics are represented in the model.

**Submission Instructions:**

Submit a single PDF and a separate code file if you performed any simulations or model evaluations.

#### **Evaluation Criteria:**

You would be evaluated on your conceptual clarity and creativity for the first question and the . rigor of ablation analysis and insightfulness from your study. Also, kindly be clear and coherent in your response.   
