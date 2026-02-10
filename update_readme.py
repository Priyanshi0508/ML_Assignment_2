import pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent
METRICS_PATH = ROOT / "model" / "saved_models" / "metrics.csv"
README_PATH = ROOT / "README.md"

if not METRICS_PATH.exists():
    print(f"Metrics file not found: {METRICS_PATH}. Run model/train_models.py first.")
    raise SystemExit(1)

metrics = pd.read_csv(METRICS_PATH)

# Build markdown table
headers = ["Model", "Accuracy", "AUC", "Precision", "Recall", "F1", "Kappa"]
md_lines = []
md_lines.append("| " + " | ".join(headers) + " |")
md_lines.append("|" + "---|" * len(headers))
for _, row in metrics.iterrows():
    vals = [str(row.get(h, "")) for h in headers]
    md_lines.append("| " + " | ".join(vals) + " |")

md_table = "\n".join(md_lines)

# Replace between markers in README
text = README_PATH.read_text(encoding="utf-8")
start_marker = "<!-- METRICS_START -->"
end_marker = "<!-- METRICS_END -->"
if start_marker in text and end_marker in text:
    before, rest = text.split(start_marker, 1)
    _, after = rest.split(end_marker, 1)
    new_text = before + start_marker + "\n\n" + md_table + "\n\n" + end_marker + after
    README_PATH.write_text(new_text, encoding="utf-8")
    print("README.md updated with latest metrics.")
else:
    print("Markers not found in README.md. No changes made.")
