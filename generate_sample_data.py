"""Generate a sample dataset and save it to data/sample_marketing_campaign.csv."""

from utils.data_generator import generate_marketing_campaign_data

if __name__ == "__main__":
    df = generate_marketing_campaign_data(n_samples=5000, treatment_effect=0.10, seed=42)
    df.to_csv("data/sample_marketing_campaign.csv", index=False)
    print(f"Saved {len(df)} rows to data/sample_marketing_campaign.csv")
    print(f"Columns: {list(df.columns)}")
    print(f"Treatment rate: {df['treatment'].mean():.2%}")
    print(f"Conversion rate: {df['conversion'].mean():.2%}")
    print(f"True ATE: {df['true_ite'].mean():.4f}")
