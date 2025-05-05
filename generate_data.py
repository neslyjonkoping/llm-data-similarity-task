import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency, ks_2samp, wasserstein_distance

# dataset generation 
np.random.seed(42)
num_samples_orig = 500
df_orig = pd.DataFrame({
    "Category1": np.random.choice(["A", "B", "C", "D", "E"], num_samples_orig, p=[0.2, 0.4, 0.2, 0.1, 0.1]),
    "Value1": np.random.normal(10, 2, num_samples_orig),
    "Value2": np.random.normal(20, 6, num_samples_orig),
})

# dataset generation with different parameters
np.random.seed(123)  

new_probs = np.random.dirichlet(np.ones(5), size=1)[0]
categories = ["A", "B", "C", "D", "E"]
num_samples_new = 1000

mean1, std1 = np.random.uniform(9, 11), np.random.uniform(1.5, 2.5)
mean2, std2 = np.random.uniform(18, 22), np.random.uniform(5, 7)

df_new = pd.DataFrame({
    "Category1": np.random.choice(categories, num_samples_new, p=new_probs),
    "Value1": np.random.normal(mean1, std1, num_samples_new),
    "Value2": np.random.normal(mean2, std2, num_samples_new),
})

df_new.to_csv("new_dataset.csv", sep=";", index=False)

# Verification
def compare_distributions(df1, df2):
    print("Category1 distribution (original vs new):")
    print(df1['Category1'].value_counts(normalize=True))
    print(df2['Category1'].value_counts(normalize=True))

    chi2, p, *_ = chi2_contingency([df1['Category1'].value_counts(), df2['Category1'].value_counts()])
    print(f"Chi-squared test for Category1: p={p:.4f}")

    for col in ["Value1", "Value2"]:
        ks_stat, ks_p = ks_2samp(df1[col], df2[col])
        wd = wasserstein_distance(df1[col], df2[col])
        print(f"\n{col}: KS p-value = {ks_p:.4f}, Wasserstein distance = {wd:.4f}")

compare_distributions(df_orig, df_new)

# Plotting
df_orig['source'] = 'original'
df_new['source'] = 'new'
combined = pd.concat([df_orig[['Value1', 'Value2', 'source']], df_new[['Value1', 'Value2', 'source']]])
combined.boxplot(column=["Value1"], by="source")
plt.title("Boxplot of Value1")
plt.show()
