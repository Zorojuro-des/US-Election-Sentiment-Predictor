import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

sns.set(style="whitegrid")

# Create a folder to save plots
plot_dir = "plots"
os.makedirs(plot_dir, exist_ok=True)

# Load CSVs
knn_df = pd.read_csv("KNNClassifierPerformanceMetrics.csv")
nn_class_df = pd.read_csv("NeuralNetoworkClassificationErrors.csv")
linreg_df = pd.read_csv("LinearRegressionErrors.csv")
nn_reg_df = pd.read_csv("NeuralNetworkRegressionErrors.csv")
poly_df = pd.read_csv("Team11_PolynomialErrors.csv")

# Clean column names
for df in [linreg_df, nn_reg_df, poly_df]:
    df.columns = df.columns.str.strip()

def plot_knn_metrics(df):
    df_melted = df.melt(id_vars="Method", var_name="Metric", value_name="Value")
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_melted, x="Metric", y="Value", hue="Method")
    plt.title("KNN Classifier Performance Metrics")
    plt.ylim(0, 1.05)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/KNN_Performance.png")
    plt.close()

def plot_nn_classifier_metrics(df):
    df_melted = df.melt(id_vars="Method", var_name="Metric", value_name="Value")
    metrics = ["Accuracy", "Avg Candidate Accuracy"]
    subset = df_melted[df_melted["Metric"].isin(metrics)]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subset, x="Metric", y="Value", hue="Method")
    plt.title("Neural Network Classifier Accuracies")
    plt.ylim(0, 105)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/NN_Classifier_Accuracies.png")
    plt.close()

def plot_regression_model(df, title, prefix):
    mse_df = df[df["Model"].str.contains("MSE", case=False)].melt(id_vars="Model", var_name="Set", value_name="MSE")
    r2_df = df[df["Model"].str.contains("R Square|R-Square", case=False)].melt(id_vars="Model", var_name="Set", value_name="R²")
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=mse_df, x="Set", y="MSE", hue="Model")
    plt.title(f"{title} - MSE")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{prefix}_MSE.png")
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.barplot(data=r2_df, x="Set", y="R²", hue="Model")
    plt.title(f"{title} - R² Score")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/{prefix}_R2.png")
    plt.close()

def comparative_classification_plot(knn_df, nn_class_df):
    knn_acc = knn_df[["Method", "Accuracy"]].copy()
    knn_acc["Model"] = "KNN"

    nn_acc = nn_class_df[["Method", "Accuracy"]].copy()
    nn_acc["Model"] = "NeuralNet"

    combined = pd.concat([knn_acc, nn_acc], ignore_index=True)

    plt.figure(figsize=(10, 6))
    sns.barplot(data=combined, x="Method", y="Accuracy", hue="Model")
    plt.title("Classification Accuracy Comparison: KNN vs NeuralNet")
    plt.ylim(0, 105)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Comparison_Classification_Accuracy.png")
    plt.close()

def comparative_regression_plot(lin_df, nn_df, poly_df, metric="MSE"):
    def melt_model(df, name, metric_type):
        sub = df[df["Model"].str.contains(metric_type, case=False)].copy()
        melted = sub.melt(id_vars="Model", var_name="Set", value_name="Value")
        melted["Model Type"] = name
        melted["Metric"] = metric_type
        return melted

    lin_melt = melt_model(lin_df, "Linear", metric)
    nn_melt = melt_model(nn_df, "NeuralNet", metric)
    poly_melt = melt_model(poly_df, "Polynomial", metric)

    full = pd.concat([lin_melt, nn_melt, poly_melt], ignore_index=True)

    plt.figure(figsize=(12, 6))
    sns.barplot(data=full, x="Set", y="Value", hue="Model Type")
    plt.title(f"Regression Model Comparison - {metric}")
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/Comparison_Regression_{metric.replace(' ', '_')}.png")
    plt.close()

if __name__ == "__main__":
    print("Saving plots for individual model performance...")

    plot_knn_metrics(knn_df)
    plot_nn_classifier_metrics(nn_class_df)

    plot_regression_model(linreg_df, "Linear Regression", "Linear")
    plot_regression_model(nn_reg_df, "Neural Network Regression", "NeuralNet")
    plot_regression_model(poly_df, "Polynomial Regression", "Polynomial")

    print("Saving comparative plots...")

    comparative_classification_plot(knn_df, nn_class_df)
    comparative_regression_plot(linreg_df, nn_reg_df, poly_df, metric="MSE")
    comparative_regression_plot(linreg_df, nn_reg_df, poly_df, metric="R Square")

    print(f"\n All plots saved to the '{plot_dir}/' directory.")
