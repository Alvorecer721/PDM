import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})

def plot_nll_distributions_ridge(results_dict, model_key, upper_quantile=1.):
    """
    Create a ridge plot of NLL distributions for each subkey in results_dict[model_key].
    
    Parameters
    ----------
    results_dict : dict
        The dictionary containing all model results (e.g., goldfish_res_greedy).
    model_key : str
        The dictionary key for the specific model 
        (e.g., 'llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60_SEQ_1984000').
    """

    # 1. Reshape your data into a DataFrame with columns: [ 'subkey', 'nll' ]
    data_records = []
    for subkey, value_dict in results_dict[model_key].items():
        scores = np.array(value_dict['NLL']['scores'])
        for score in scores:
            data_records.append({'subkey': subkey, 'Negative Log Likelihood': score})
    
    df = pd.DataFrame(data_records)

    # (Optional) Clip outliers or large values
    lower = df["Negative Log Likelihood"].quantile(0.00)
    upper = df["Negative Log Likelihood"].quantile(upper_quantile)
    df["Trimmed Negative Log Likelihood"] = df["Negative Log Likelihood"].clip(lower, upper)

    # 2. Initialize a palette and a FacetGrid
    unique_subkeys = df["subkey"].unique()
    pal = sns.cubehelix_palette(len(unique_subkeys), rot=-.25, light=.7)
    g = sns.FacetGrid(
        df, 
        row="subkey",                     # each subkey in its own row
        hue="subkey",                     # color by subkey
        aspect=20,                        # make plots much wider than tall
        height=.5,                        # the vertical height of each plot
        palette=pal
    )

    # 3. Plot the ridgeline KDEs (filled, then outline)
    g.map(
        sns.kdeplot, 
        "Trimmed Negative Log Likelihood",
        bw_adjust=.5,
        clip_on=False,
        fill=True,
        alpha=1, 
        linewidth=1.5
    )

    g.map(
        sns.kdeplot, 
        "Trimmed Negative Log Likelihood", 
        clip_on=False, 
        color="w", 
        lw=2, 
        bw_adjust=.5
    )

    # 4. Reference line at y=0 (for each row)
    g.refline(y=0, linewidth=2, linestyle="-", color=None, clip_on=False)

    # 5. Define and apply a small labeling function to place text within each subplot
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .3, label, 
                fontweight="bold", 
                color=color, 
                ha="left", 
                va="center", 
                transform=ax.transAxes)

    # Map the labeling function to one of the variables (not the trimmed one!)
    g.map(label, "Negative Log Likelihood")

    # 6. Adjust the subplot spacing so that subplots overlap
    g.figure.subplots_adjust(hspace=-.05)

    # 7. Remove or simplify unneeded axis details
    g.set_titles("")
    g.set(yticks=[], ylabel="")
    g.despine(bottom=True, left=True)

    # --- ADDING TITLE AND AXIS LABELS ---
    # 8. Set a main title for the entire figure
    g.fig.suptitle(
        model_key, 
        x=0.5,              # center the title
        y=1.03,             # adjust if it overlaps the topmost subplot
        fontsize=10
    )
    
    # 9. Optionally adjust the top margin to ensure the title fits
    g.figure.subplots_adjust(
        hspace=-0.25,      # Increase overlap between subplots
        left=0.1,          # Increase left margin for labels
        right=0.9,         # Adjust right margin
        top=0.95,          # Adjust top margin for title
        bottom=0.1         # Adjust bottom margin
    )
    
    # 10. Set the x-axis label for all facets
    g.set_axis_labels("Trimmed Negative Log Likelihood", "")

    plt.show()
