import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# -------------------------------
# 1. Create SQLite Database
# -------------------------------

conn = sqlite3.connect("university.db")
cursor = conn.cursor()

# Create Tables
cursor.execute("""
CREATE TABLE IF NOT EXISTS students (
    student_id INTEGER PRIMARY KEY,
    name TEXT,
    department TEXT,
    year INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS subjects (
    subject_id INTEGER PRIMARY KEY,
    subject_name TEXT,
    department TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS marks (
    student_id INTEGER,
    subject_id INTEGER,
    marks INTEGER
)
""")

# -------------------------------
# 2. Insert Sample Data
# -------------------------------

departments = ["CSE", "ECE", "MECH"]

# Insert students
for i in range(1, 101):
    cursor.execute("INSERT INTO students VALUES (?, ?, ?, ?)",
                   (i, f"Student_{i}", np.random.choice(departments), np.random.randint(1,5)))

# Insert subjects
subjects_data = [
    (1, "Data Structures", "CSE"),
    (2, "Signals", "ECE"),
    (3, "Thermodynamics", "MECH"),
    (4, "Algorithms", "CSE"),
    (5, "Microprocessors", "ECE"),
    (6, "Machine Design", "MECH")
]

cursor.executemany("INSERT INTO subjects VALUES (?, ?, ?)", subjects_data)

# Insert marks
for student_id in range(1, 101):
    for subject_id in range(1, 7):
        cursor.execute("INSERT INTO marks VALUES (?, ?, ?)",
                       (student_id, subject_id, np.random.randint(40, 100)))

conn.commit()

# -------------------------------
# 3. JOIN Strategy
# -------------------------------

query = """
SELECT s.student_id, s.name, s.department,
       sub.subject_name, m.marks
FROM students s
JOIN marks m ON s.student_id = m.student_id
JOIN subjects sub ON sub.subject_id = m.subject_id
"""

df = pd.read_sql_query(query, conn)

print("\nSample Joined Data:")
print(df.head())

# -------------------------------
# 4. Central Tendency Analysis
# -------------------------------

dept_mean = df.groupby("department")["marks"].mean()
print("\nDepartment Mean Marks:")
print(dept_mean)

best_dept = dept_mean.idxmax()
print("\nBest Performing Department:", best_dept)

# -------------------------------
# 5. Dispersion Analysis
# -------------------------------

dept_std = df.groupby("department")["marks"].std()
print("\nDepartment Std Deviation:")
print(dept_std)

max_variation = dept_std.idxmax()
print("\nDepartment with Maximum Variation:", max_variation)

# -------------------------------
# 6. Skewness Observation
# -------------------------------

print("\nSkewness by Subject:")
for subject in df["subject_name"].unique():
    subject_marks = df[df["subject_name"] == subject]["marks"]
    print(subject, ":", round(skew(subject_marks), 2))

# -------------------------------
# 7. Top 5% Students
# -------------------------------

student_avg = df.groupby(["student_id", "name"])["marks"].mean().reset_index()

threshold = np.percentile(student_avg["marks"], 95)

top_students = student_avg[student_avg["marks"] >= threshold]

print("\nTop 5% Students:")
print(top_students)

# -------------------------------
# 8. Z-score Anomaly Detection
# -------------------------------

overall_mean = df["marks"].mean()
overall_std = df["marks"].std()

df["z_score"] = (df["marks"] - overall_mean) / overall_std

abnormal = df[np.abs(df["z_score"]) > 2]

print("\nAbnormal Performances (|Z| > 2):")
print(abnormal.head())

# -------------------------------
# 9. Percentage Above 1 Std Dev
# -------------------------------

above_one_std = df[df["marks"] > overall_mean + overall_std]
percentage = (len(above_one_std) / len(df)) * 100

print("\nPercentage Above 1 Standard Deviation:", round(percentage,2), "%")

# -------------------------------
# 10. Distribution Plot
# -------------------------------

plt.hist(df["marks"], bins=20)
plt.title("Overall Marks Distribution")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

conn.close()