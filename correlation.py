import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from questions import questions_list

def correlation_analysis(df, question_index, output_dir="outputs/feature_correlation", save_name="correlation_heatmap.png"):
  label = questions_list[question_index]["label"]
  output_path = Path(output_dir)
  output_path.mkdir(parents=True, exist_ok=True)

  correlation_matrix = df.corr(numeric_only=True)

  plt.figure(figsize=(24, 22))
  ax = sns.heatmap(
    correlation_matrix,
    annot=False,
    cmap="coolwarm",
    xticklabels=True,
    yticklabels=True
  )
  plt.title("Feature Correlation Heatmap")
  ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
  ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

  plt.tight_layout()
  plt.savefig(output_path / save_name, dpi=300)
  plt.close()

  correlations = correlation_matrix[label].drop(label)
  relevant_corr = correlations[correlations.abs() > 0.1]
  correlated_features = relevant_corr.index.tolist()

  correlations = correlation_matrix[label].drop(label)
  correlated_features = correlations[correlations.abs() > 0.1].index.tolist()

  report_path = output_path / f"question_{question_index+1}_correlations.txt"
  with open(report_path, "w") as f:
    f.write(f"Label: {label}\n")
    f.write("Correlated features (|corr| > 0.1):\n\n")
    for feature, value in relevant_corr.items():
      f.write(f"- {feature}: {value:.4f}\n")

  return correlated_features