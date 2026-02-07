"""Generate and save sample outreach data for local development."""

import os
from utils.data_generator import generate_outreach_data

df = generate_outreach_data(n_records=8000, seed=42)
os.makedirs("data", exist_ok=True)
df.to_csv("data/sample_outreach_data.csv", index=False)

print(f"Generated {len(df)} records")
print(f"Suggestion acceptance rate: {df['Suggestion_Accepted'].mean():.2%}")
print(f"Mean Incremental TRX: {df['Incremental_TRX'].mean():.3f}")
print(f"Mean Incremental NBRX: {df['Incremental_NBRX'].mean():.3f}")
print(f"\nSuggestion Type distribution:")
print(df["Suggestion_Type"].value_counts())
print(f"\nAction Type distribution:")
print(df["Action_Type"].value_counts())
