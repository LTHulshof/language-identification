import matplotlib.pyplot as plt
import numpy as np

# Data
languages = ["deu", "ell", "eng", "nld", "spa", "unk"]

roberta_easy = [0.4206, 0.8929, 0.2457, 0.4623, 0.5702, 0.5854]
roberta_medium = [0.6534, 0.9574, 0.6862, 0.7123, 0.7546, 0.6417]
roberta_hard = [0.8270, 0.9572, 0.8085, 0.8035, 0.9110, 0.5916]

# Group setup
x = np.arange(len(languages))  # positions for each language
width = 0.25  # slim bars

fig, ax = plt.subplots(figsize=(10, 6))

# Bars
ax.bar(x - width, roberta_easy, width, label="RoBERTa Easy", color="#ff6666")
ax.bar(x, roberta_medium, width, label="RoBERTa Medium", color="#ffcc66")
ax.bar(x + width, roberta_hard, width, label="RoBERTa Hard", color="#66b266")

# Labels and styling
ax.set_xlabel("Language", fontsize=12)
ax.set_ylabel("Accuracy", fontsize=12)
ax.set_title("Mixed data language accuracy", fontsize=14, weight="bold")
ax.set_xticks(x)
ax.set_xticklabels(languages)
ax.legend(title="Model")
ax.set_ylim(0, 1.05)
ax.grid(axis="y", linestyle="--", alpha=0.6)

plt.tight_layout()
plt.show()