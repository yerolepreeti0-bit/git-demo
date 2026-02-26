import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import random

# 1. Generate Dataset (1500 records)
np.random.seed(42)
n = 1500
mean_weight = 100
std_dev = 5

data = pd.DataFrame({
    "batch_id": np.random.randint(1, 21, n),
    "part_id": range(1, n + 1),
    "weight": np.random.normal(mean_weight, std_dev, n),
    "machine_id": np.random.randint(1, 6, n),
    "shift": np.random.choice(["Morning", "Evening", "Night"], n)
})
print("\nFirst 5 Records:")
print(data.head())

# 2. Normal Distribution Check
mean_val = data["weight"].mean()
median_val = data["weight"].median()

print("\nMean:", round(mean_val,2))
print("Median:", round(median_val,2))

plt.hist(data["weight"], bins=30, density=True)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean_val, std_dev)
plt.plot(x, p)
plt.title("Histogram with Normal Curve")
plt.show()

# 3. Probability Calculations
# P(weight < 90g)
z1 = (90 - mean_weight) / std_dev
prob_less_90 = norm.cdf(z1)
print("\nP(weight < 90g):", round(prob_less_90,4))

# P(95g < weight < 105g)
z2 = (95 - mean_weight) / std_dev
z3 = (105 - mean_weight) / std_dev
prob_between = norm.cdf(z3) - norm.cdf(z2)
print("P(95g < weight < 105g):", round(prob_between,4))

# 4. Central Limit Theorem
sample_means = []

for i in range(100):
    sample = data["weight"].sample(40)
    sample_means.append(sample.mean())

plt.hist(sample_means, bins=20, density=True)
plt.title("Sampling Distribution (CLT)")
plt.show()
print("\nMean of Sample Means:", round(np.mean(sample_means),2))

# 5. Z-score for Defect Detection
data["z_score"] = (data["weight"] - mean_val) / std_dev

defective_parts = data[np.abs(data["z_score"]) > 2.5]

print("\nNumber of Defective Parts:", len(defective_parts))
print("\nSample Defective Parts:")
print(defective_parts.head())

# 6. Process Stability Check
if len(defective_parts) < 0.01 * n:
    print("\nProcess is Stable ")
else:
    print("\nProcess Needs Investigation ")