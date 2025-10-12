import argparse
import os
import sys
import urllib.request
from typing import Dict, List, Tuple


MODEL_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin"
MODEL_PATH = os.path.join(os.path.dirname(__file__), "lid.176.bin")

# Dataset files default to the local wili_subset directory
DATA_DIR = os.path.join(os.path.dirname(__file__), "wili_subset")

# Dataset labels (ISO-639-3) to fastText labels (ISO-639-1-ish) mapping
DATASET_TO_FASTTEXT: Dict[str, str] = {
	"nld": "nl",
	"deu": "de",
	"eng": "en",
	"ell": "el",
	"spa": "es",
}

TARGET_FT_LABELS: List[str] = [
	DATASET_TO_FASTTEXT[code] for code in ["nld", "deu", "eng", "ell", "spa"]
]


def download_model_if_missing(model_path: str = MODEL_PATH) -> None:
	if os.path.exists(model_path):
		return
	print(f"Downloading FastText LID model to {model_path} ...", flush=True)
	os.makedirs(os.path.dirname(model_path), exist_ok=True)
	urllib.request.urlretrieve(MODEL_URL, model_path)
	print("Download complete.", flush=True)


def load_fasttext_model(model_path: str = MODEL_PATH):
	try:
		import fasttext  # type: ignore
	except Exception as exc:  # pragma: no cover
		print(
			"Missing dependency: fasttext. Install with: pip install fasttext-wheel or pip install fasttext==0.9.2",
			file=sys.stderr,
		)
		raise exc

	# Prefer the canonical API
	if hasattr(fasttext, "load_model"):
		return fasttext.load_model(model_path)

	# Try submodule/class fallbacks found in some builds
	try:
		from fasttext import FastText  # type: ignore
		if hasattr(FastText, "load_model"):
			return FastText.load_model(model_path)
		if hasattr(FastText, "_load_model"):
			return FastText._load_model(model_path)  # type: ignore[attr-defined]
	except Exception:
		pass

	raise RuntimeError(
		"fasttext module does not expose load_model. Please reinstall: pip install fasttext==0.9.2"
	)


def read_lines(path: str) -> List[str]:
	with open(path, "r", encoding="utf-8") as f:
		return [line.rstrip("\n") for line in f]


def load_split(
	data_dir: str,
	split: str,
	use_replaced: bool,
) -> Tuple[List[str], List[str]]:
	assert split in {"train", "test"}
	postfix = "_replaced.txt" if use_replaced else ".txt"
	x_path = os.path.join(data_dir, f"x_{split}{postfix}")
	y_path = os.path.join(data_dir, f"y_{split}.txt")
	if not os.path.exists(x_path):
		raise FileNotFoundError(f"Missing input file: {x_path}")
	if not os.path.exists(y_path):
		raise FileNotFoundError(f"Missing label file: {y_path}")
	texts = read_lines(x_path)
	labels = read_lines(y_path)
	# Align lengths defensively
	n = min(len(texts), len(labels))
	if len(texts) != len(labels):
		print(
			f"Warning: length mismatch (texts={len(texts)}, labels={len(labels)}); trimming to {n}",
			file=sys.stderr,
		)
	return texts[:n], labels[:n]


def map_dataset_label_to_ft(label3: str) -> str:
	if label3 not in DATASET_TO_FASTTEXT:
		raise KeyError(f"Unexpected dataset label: {label3}")
	return DATASET_TO_FASTTEXT[label3]


def predict_restricted(model, text: str, allowed_ft_labels: List[str]) -> Tuple[str, float]:
	# Ask for many labels to ensure the allowed one is surfaced; k set conservatively high
	labels, probs = model.predict(text, k=200)
	# fastText returns labels like __label__xx
	for lbl, p in zip(labels, probs):
		clean = lbl.replace("__label__", "")
		if clean in allowed_ft_labels:
			return clean, float(p)
	# If none of the allowed labels were predicted, fall back to the top label
	clean = labels[0].replace("__label__", "")
	return clean, float(probs[0])


def compute_confusion(
	true_labels3: List[str],
	pred_labels_ft: List[str],
):
	# Fix label order for display
	order3 = ["nld", "deu", "eng", "ell", "spa"]
	order_ft = [DATASET_TO_FASTTEXT[l] for l in order3]
	index3 = {l: i for i, l in enumerate(order3)}
	index_ft = {l: i for i, l in enumerate(order_ft)}

	# Initialize matrix
	conf = [[0 for _ in order3] for _ in order3]
	correct = 0
	for y3, yhat_ft in zip(true_labels3, pred_labels_ft):
		i = index3.get(y3, None)
		j = index_ft.get(yhat_ft, None)
		if i is None or j is None:
			continue
		conf[i][j] += 1
		if order_ft[j] == DATASET_TO_FASTTEXT[y3]:
			correct += 1
	return conf, correct, len(true_labels3)


def print_confusion(conf, row_labels: List[str], col_labels: List[str]) -> None:
	# Pretty print a small confusion matrix
	col_width = max(5, max(len(x) for x in row_labels + col_labels))
	header = "".ljust(col_width) + " " + " ".join(lbl.ljust(col_width) for lbl in col_labels)
	print(header)
	for r_label, row in zip(row_labels, conf):
		row_str = " ".join(str(v).ljust(col_width) for v in row)
		print(r_label.ljust(col_width) + " " + row_str)


def evaluate(
	split: str,
	use_replaced: bool,
	data_dir: str = DATA_DIR,
	model_path: str = MODEL_PATH,
) -> None:
	texts, labels3 = load_split(data_dir, split, use_replaced)
	download_model_if_missing(model_path)
	model = load_fasttext_model(model_path)

	true_ft = [map_dataset_label_to_ft(y3) for y3 in labels3]
	preds_ft: List[str] = []
	probs: List[float] = []
	for t in texts:
		pred, p = predict_restricted(model, t, TARGET_FT_LABELS)
		preds_ft.append(pred)
		probs.append(p)

	conf, correct, total = compute_confusion(labels3, preds_ft)
	acc = correct / total if total else 0.0
	print(f"Samples: {total}")
	print(f"Accuracy (restricted to 5 languages): {acc:.4f}")
	print()
	print("Confusion matrix (rows=true ISO-639-3, cols=pred fastText codes):")
	row_labels = ["nld", "deu", "eng", "ell", "spa"]
	col_labels = [DATASET_TO_FASTTEXT[l] for l in row_labels]
	print_confusion(conf, row_labels, col_labels)


def main() -> None:
	parser = argparse.ArgumentParser(description="Evaluate FastText LID (restricted to 5 languages)")
	parser.add_argument("--split", choices=["train", "test"], default="test")
	parser.add_argument("--replaced", action="store_true", help="Use *_replaced.txt for inputs")
	parser.add_argument("--data-dir", default=DATA_DIR)
	parser.add_argument("--model-path", default=MODEL_PATH)
	args = parser.parse_args()

	evaluate(split=args.split, use_replaced=args.replaced, data_dir=args.data_dir, model_path=args.model_path)


if __name__ == "__main__":
	main()


