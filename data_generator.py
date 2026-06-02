import pandas as pd
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Simulate a data ingestion pipeline.")
parser.add_argument('--corrupt', action='store_true', help='Inject severe noise into the dataset')
args = parser.parse_args()

# Generate synthetic dataset (Predicting Salary based on Experience and Education)
np.random.seed(42)
n_samples = 1000
experience = np.random.uniform(0, 40, n_samples)
education = np.random.randint(12, 22, n_samples)

if args.corrupt:
    # DISASTER: Someone updated the pipeline and injected massive noise/errors!
    noise = np.random.normal(0, 100000, n_samples)
    salary = 30000 + (experience * 2000) + (education * 3000) + noise
    print("❌ CRITICAL ALARM: Corrupted dataset generated (Simulating a pipeline failure).")
else:
    # Clean, healthy data
    noise = np.random.normal(0, 5000, n_samples)
    salary = 30000 + (experience * 2000) + (education * 3000) + noise
    print("✅ SUCCESS: Clean dataset generated (V1).")

df = pd.DataFrame({'Experience': experience, 'Education': education, 'Salary': salary})
df.to_csv('dataset.csv', index=False)
