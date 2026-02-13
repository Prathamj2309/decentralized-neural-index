import pandas as pd
import matplotlib.pyplot as plt
import os

RESULTS_DIR = "experiments/msmarco"
ABLATION_FILE = os.path.join(RESULTS_DIR, "ablation_results.csv")
STAT_FILE = os.path.join(RESULTS_DIR, "statistical_results.csv")

os.makedirs(RESULTS_DIR, exist_ok=True)

# =============================
# LOAD ABLATION RESULTS
# =============================

df_ablation = pd.read_csv(ABLATION_FILE)

# Filter only normalized + stopwords=True for clean alpha curve
df_clean = df_ablation[
    (df_ablation["use_stopwords"] == True) &
    (df_ablation["use_normalization"] == True)
]

# =============================
# PLOT 1 — Alpha vs NDCG
# =============================

plt.figure()
plt.plot(df_clean["alpha"], df_clean["mean_ndcg"], marker='o')
plt.xlabel("Alpha (Semantic Weight)")
plt.ylabel("Mean NDCG@10")
plt.title("Alpha Sensitivity (NDCG)")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "alpha_vs_ndcg.png"))
plt.close()

# =============================
# PLOT 2 — Alpha vs MRR
# =============================

plt.figure()
plt.plot(df_clean["alpha"], df_clean["mean_mrr"], marker='o')
plt.xlabel("Alpha (Semantic Weight)")
plt.ylabel("Mean MRR@10")
plt.title("Alpha Sensitivity (MRR)")
plt.grid(True)
plt.savefig(os.path.join(RESULTS_DIR, "alpha_vs_mrr.png"))
plt.close()

# =============================
# PLOT 3 — Float vs Int8 Comparison
# =============================

df_stat = pd.read_csv(STAT_FILE)

float_ndcg = df_stat[
    (df_stat["metric"] == "NDCG") &
    (df_stat["system_A"] == "Float")
]["mean_A"].values[-1]

int8_ndcg = df_stat[
    (df_stat["metric"] == "NDCG") &
    (df_stat["system_A"] == "Float")
]["mean_B"].values[-1]

plt.figure()
plt.bar(["Float32", "Int8"], [float_ndcg, int8_ndcg])
plt.ylabel("Mean NDCG@10")
plt.title("Float vs Int8 Retrieval Performance")
plt.savefig(os.path.join(RESULTS_DIR, "float_vs_int8.png"))
plt.close()

print("Plots saved to experiments/msmarco/")
