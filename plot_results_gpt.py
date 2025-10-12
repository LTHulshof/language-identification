import matplotlib.pyplot as plt
import numpy as np

# Accuracy results
results = {
    "Gemini": 0.3117,
    "FastText": 0.8531,
    "Google Translate": 0.7298,
    "RoBERTa-Easy": 0.4925,
    "RoBERTa-Medium": 0.8039,
    "RoBERTa-Hard": 0.8081
}

# Extract names and accuracies
models = list(results.keys())
accuracies = np.array(list(results.values()))

# Create color gradient from red (low acc) to green (high acc)
colors = plt.cm.RdYlGn(accuracies)

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=colors, edgecolor="black")

ax.set_title("GPT data model accuracy")
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_xticklabels(models, rotation=45, ha="right")

# Add colorbar (link it to current axes)
sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=1))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)

plt.tight_layout()
plt.show()