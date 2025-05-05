# llm-data-similarity-task
Dataset generation and similarity verification task 
# Task: LLM Dataset Generation and Verification

This repository contains the solution to the coding task. The goal is to generate a new dataset that is *similar* to an original synthetic dataset, using **different sampling parameters**, and then **verify the similarity** between them.

##  Task Overview

> "Use the provided code to generate a synthetic dataset, then generate a new similar dataset using different sampling parameters. Verify that the new data resembles the original."

---

##  Reasoning Behind the Approach

1. **Original Structure**:
   - `Category1`: Categorical with non-uniform probabilities.
   - `Value1`, `Value2`: Continuous variables drawn from normal distributions.

2. **New Generation Strategy**:
   - Used a **Dirichlet distribution** to randomly generate category probabilities.
   - Chose new **means and standard deviations** for `Value1` and `Value2` within realistic ranges.
   - Increased sample size to 1000 for better statistical comparisons.

3. **Verification Methods**:
   - **Chi-Squared Test** for categorical similarity.
   - **Kolmogorov-Smirnov Test** and **Wasserstein Distance** for continuous variables.
   - **Visualizations**: Boxplots for intuitive understanding.

Results

Chi-squared test for Category1: p=0.0181

Value1: KS p-value = 0.0099, Wasserstein distance = 0.3471

Value2: KS p-value = 0.0462, Wasserstein distance = 0.6991
---

##  Usage

Clone the repo and run the script:

```bash
git clone https://github.com/yourusername/llm-data-similarity-task.git
cd llm-data-similarity-task
python generate_data.py




