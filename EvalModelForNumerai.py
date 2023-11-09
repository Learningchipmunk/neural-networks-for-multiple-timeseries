import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

# Numerai's primary scoring metric
def numerai_corr(preds, target):
    # rank (keeping ties) then gaussianize predictions to standardize prediction distributions
    ranked_preds = (preds.rank(method="average").values - 0.5) / preds.count()
    gauss_ranked_preds = stats.norm.ppf(ranked_preds)
    # center targets around 0
    centered_target = target - target.mean()
    # raise both preds and target to the power of 1.5 to accentuate the tails
    preds_p15 = np.sign(gauss_ranked_preds) * np.abs(gauss_ranked_preds) ** 1.5
    target_p15 = np.sign(centered_target) * np.abs(centered_target) ** 1.5
    # finally return the Pearson correlation
    return np.corrcoef(preds_p15, target_p15)[0, 1]

# Compute performance metrics, a vectorized verion of what is found here https://github.com/numerai/example-scripts/blob/master/hello_numerai.ipynb 
def compute_numerai_metrics(per_era_corr, method_names=["Value"]):
    corr_mean = pd.Series(per_era_corr.mean(axis=0))
    corr_min = pd.Series(per_era_corr.min(axis=0))
    corr_max = pd.Series(per_era_corr.max(axis=0))
    corr_std = pd.Series(per_era_corr.std(axis=0, ddof=0))
    corr_sharpe = pd.Series(corr_mean / corr_std)
    max_drawdown = (per_era_corr.cumsum(axis=0).expanding(min_periods=1).max() - per_era_corr.cumsum(axis=0)).max(axis=0)

    return pd.DataFrame({
        "mean": corr_mean.tolist(),
        "min": corr_min.tolist(),
        "max": corr_max.tolist(),
        "std": corr_std.tolist(),
        "sharpe": corr_sharpe.tolist(),
        "max_drawdown": max_drawdown.tolist()
    }, index=method_names).T

def PlotCorr(per_era_corr, kind='line', title="Validation Correlation", era_per_tick=20, fontsize=12, add_mean_to_plot=False, save_fig=False, save_path="valid_corr.png"):
    # Converting to DataFrame for vectorization
    per_era_corr = pd.DataFrame(per_era_corr)

    # Plotting the correlation per era
    ax = per_era_corr.plot(kind=kind, title=title, figsize=(16, 8), xticks=per_era_corr.index.to_list()[::era_per_tick], snap=False)

    # Adding the mean of the correlation per era to plot if requested by user
    if(add_mean_to_plot):
        means = pd.Series(per_era_corr.mean(axis=0))
        colors = ['r', 'b', 'c', 'm', 'y', 'k', 'g']
        i=0
        for mean, name in zip(means, per_era_corr.columns):
            ax.axhline(y=mean, color=colors[i], linestyle='-.', label=f"mean of {name}: {mean:.4f}")
            i = (i+1)%len(colors)

    # Adding legend and setting fontsize
    ax.legend(fontsize=fontsize, loc="upper left")

    # Saving the figure if requested by user
    if save_fig:
        plt.savefig(save_path)
    else:
        plt.show()

def ComputePerEraCorr(df, prediction_names, target_name, save_df=False, save_path="per_era_corr.csv"):
    per_era_corr = pd.DataFrame()
    
    for pred_name in prediction_names:
        era_df = df.groupby("era_int").apply(lambda x: numerai_corr(x[pred_name], x[target_name])).sort_index().rename(pred_name.lower() + "2targ_model_corr")
        per_era_corr = pd.concat([per_era_corr, era_df], axis=1)

    if save_df:
        per_era_corr.to_csv(save_path)

    return per_era_corr


class EvaluateModelForNumerai:
    def __init__(self, target_name="target", mymodel_name='prediction', save_folder_path="", save_metric_df=False, save_plots=False):
        self.target_name = target_name
        self.mymodel_name = mymodel_name
        self.save_folder_path = save_folder_path
        self.save_metric_df = save_metric_df
        self.save_plots = save_plots

    def assess_model_performance(self, df, prediction_names, model_names, add_means_to_plot=False, plot_cor_my_model=False, file_name_png="all_models_cumsum.png", file_name_csv="metrics_numerai.csv"):
        # Computes the per-era correlation for each prediction in prediction_names
        per_era_corr = ComputePerEraCorr(df, prediction_names, target_name=self.target_name, save_df=False)

        print(per_era_corr.columns)

        # Plot the per-era correlation
        if plot_cor_my_model:
            PlotCorr(per_era_corr[per_era_corr.columns[0]], kind='line', title="Validation Correlation of My Model", era_per_tick=20, fontsize=12, add_mean_to_plot=add_means_to_plot, save_fig=self.save_plots, save_path=self.save_folder_path + "My_Model_valid_corr.png")

        # Plot the cumulative per-era correlation for all predictions
        PlotCorr(per_era_corr.cumsum(axis=0), kind='line', title="Cumulative Validation Correlation", era_per_tick=20, fontsize=12, add_mean_to_plot=add_means_to_plot, save_fig=self.save_plots, save_path=self.save_folder_path + file_name_png)

        ## `My Model` performance compared to benchmarks
        # mean corr is the most important metric
        # Sharpe is mean/std, it is the risk of the model, the higher the less risky
        # Max drawdown is the max loss from a peak to a trough of a portfolio, before a new peak is attained, the lower the better
        metric_df = compute_numerai_metrics(per_era_corr, model_names)

        # Saves the metrics as a csv file if requested by user
        if self.save_metric_df:
            metric_df.to_csv(self.save_folder_path + file_name_csv)

        # Deletes the per_era_corr dataframe to free up memory
        del per_era_corr
        _ = gc.collect()

        return metric_df
