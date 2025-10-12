import matplotlib.pyplot as plt

easy = [0.9611, 0.9553, 0.9503, 0.9420, 0.9359, 0.9302, 0.9244, 0.9234, 0.9108, 0.9092, 0.9011]
medium = [0.9502, 0.9485, 0.9462, 0.9435, 0.9402, 0.9389, 0.9363, 0.9363, 0.9332, 0.9318, 0.9288]
hard = [0.9441, 0.9432, 0.9414, 0.9405, 0.9379, 0.9369, 0.9343, 0.9337, 0.9329, 0.9318, 0.9282]
replacement = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

plt.figure(figsize=(8, 5))

plt.plot(
    replacement,
    easy,
    marker="o",
    markersize=4,
    linestyle="-",
    color="green",
    linewidth=1.8,
    label="Easy"
)

plt.plot(
    replacement,
    medium,
    marker="o",
    markersize=4,
    linestyle="-",
    color="red",
    linewidth=1.8,
    label="Medium"
)

plt.plot(
    replacement,
    hard,
    marker="o",
    markersize=4,
    linestyle="-",
    color="blue",
    linewidth=1.8,
    label="Hard"
)

plt.title("XLM-RoBERTa Span Frequency Accuracy")
plt.xlabel("Span Replacement Ratio")
plt.ylabel("Accuracy")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend(["XLM-RoBERTa-easy", "XLM-RoBERTa-medium", "XLM-RoBERTa-hard"])
plt.tight_layout()
plt.show()