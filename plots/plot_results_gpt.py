import matplotlib.pyplot as plt
import numpy as np

# Accuracy results
results_token = {
    "Gemini 2.5 pro": 0.3117,
    "FastText": 0.8531,
    "Google Translate": 0.7298,
    "RoBERTa-Easy": 0.4925,
    "RoBERTa-Medium": 0.8039,
    "RoBERTa-Hard": 0.8081
}

models = list(results_token.keys())
accuracies = np.array(list(results_token.values()))

vmin, vmax = 0, 1
norm = plt.Normalize(vmin=vmin, vmax=vmax)
cmap = plt.cm.managua
colors = cmap(norm(accuracies))  

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(models, accuracies, color=colors)

ax.set_title("GPT Data Model Accuracy")
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_ylim(0, 1)
ax.set_xticklabels(models, rotation=45, ha="right")

sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax)

plt.tight_layout()
plt.show()