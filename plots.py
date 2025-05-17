import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_results_boxplots(results_root="results_bkp2", output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through each question folder (e.g., question1, question2, ...)
    for question_folder in os.listdir(results_root):
        question_path = os.path.join(results_root, question_folder)
        if not os.path.isdir(question_path):
            continue

        print(f"📊 Processing: {question_folder}")

        # Dictionary to store metric data: { metric_name: [ {Model: ..., Value: ...}, ... ] }
        metric_data = {}

        # Process each model's result CSV in this question folder
        for file in os.listdir(question_path):
            if not file.endswith("_results.csv"):
                continue

            model_name = file.replace("_results.csv", "")
            file_path = os.path.join(question_path, file)
            df = pd.read_csv(file_path)

            for column in df.columns:
                if column not in metric_data:
                    metric_data[column] = []

                for value in df[column]:
                    metric_data[column].append({
                        "Model": model_name,
                        "Value": value
                    })

        # Generate a box plot for each metric
        for metric, records in metric_data.items():
            plot_df = pd.DataFrame(records)

            plt.figure(figsize=(10, 6))
            sns.boxplot(data=plot_df, x="Model", y="Value", palette="pastel")
            plt.title(f"{question_folder} - {metric}", fontsize=14)
            plt.xlabel("Model", fontsize=12)
            plt.ylabel(metric, fontsize=12)
            plt.xticks(rotation=45)
            plt.tight_layout()

            filename = f"{question_folder}_{metric}_boxplot.png".replace(" ", "_").lower()
            plot_path = os.path.join(output_dir, filename)
            plt.savefig(plot_path)
            plt.close()
            print(f"✅ Saved: {plot_path}")

if __name__ == "__main__":
    plot_results_boxplots()
