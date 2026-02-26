import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, skew
import sqlite3

# ---------------- MANUFACTURING ANALYSIS ----------------
np.random.seed(42)
n = 1500
mean_w = 100
std_w = 5
data = pd.DataFrame({
    "batch_id": np.random.randint(1, 21, n),
    "part_id": range(1, n + 1),
    "weight": np.random.normal(mean_w, std_w, n),
    "machine_id": np.random.randint(1, 6, n),
    "shift": np.random.choice(["Morning", "Evening", "Night"], n)
})

print("\nFirst 5 Manufacturing Records:")
print(data.head())

mean_val = data["weight"].mean()
median_val = data["weight"].median()

print("\nMean:", round(mean_val,2))
print("Median:", round(median_val,2))

plt.hist(data["weight"], bins=30, density=True)
x = np.linspace(data["weight"].min(), data["weight"].max(), 100)
plt.plot(x, norm.pdf(x, mean_w, std_w))
plt.title("Manufacturing Distribution")
plt.show()

print("\nP(weight < 90):", round(norm.cdf((90-mean_w)/std_w),4))
print("P(95 < weight <105):", round(norm.cdf(1)-norm.cdf(-1),4))

sample_means = [data["weight"].sample(40).mean() for _ in range(100)]

plt.hist(sample_means, bins=20)
plt.title("CLT - Sampling Distribution")
plt.show()

print("\nMean of Sample Means:", round(np.mean(sample_means),2))

data["z_score"] = (data["weight"] - mean_val) / std_w
defective = data[np.abs(data["z_score"]) > 2.5]

print("\nTotal Defective Parts:", len(defective))
print(defective.head())

# ---------------- UNIVERSITY ANALYSIS ----------------
conn = sqlite3.connect(":memory:")
cur = conn.cursor()

cur.execute("CREATE TABLE marks(student INT, dept TEXT, subject TEXT, marks INT)")
depts = ["CSE", "ECE", "MECH"]
# Insert Marks
for s in range(1, 101):
    for sub in ["Data Structures", "Signals", "hermodynamics","Algorithms","Microprocessors"]:
        cur.execute("INSERT INTO marks VALUES(?,?,?,?)",
                    (s,
                     np.random.choice(depts),
                     sub,
                     np.random.randint(40,100)))
conn.commit()

df = pd.read_sql("SELECT * FROM marks", conn)

print("\nSample University Data:")
print(df.head())

# 1️. Best Performing Department
dept_mean = df.groupby("dept")["marks"].mean().round(2)
print("\nDepartment Mean Marks:")
print(dept_mean)

best_dept = dept_mean.idxmax()
print("\nBest Performing Department:", best_dept)

# 2️. Department with Maximum Variation
dept_std = df.groupby("dept")["marks"].std().round(2)
print("\nDepartment Standard Deviation:")
print(dept_std)

max_variation = dept_std.idxmax()
print("\nDepartment with Maximum Variation:", max_variation)

# 3️. Skewness by Subject
print("\nSkewness by Subject:")
for subject in df["subject"].unique():
    subject_marks = df[df["subject"] == subject]["marks"]
    print(subject, ":", round(skew(subject_marks), 2))

# 4️. Top 5% Students
student_avg = df.groupby("student")["marks"].mean().reset_index()

threshold = np.percentile(student_avg["marks"], 95)
top_students = student_avg[student_avg["marks"] >= threshold]

print("\nTop 5% Students:")
print(top_students.sort_values(by="marks", ascending=False))

# 5️. Z-Score Anomalies
overall_mean = df["marks"].mean()
overall_std = df["marks"].std()

df["z_score"] = (df["marks"] - overall_mean) / overall_std
abnormal = df[np.abs(df["z_score"]) > 2]

print("\nAbnormal Performances (|Z| > 2):")
print(abnormal)

plt.hist(df["marks"], bins=20)
plt.title("Marks Distribution")
plt.show()

conn.close()