import matplotlib.pyplot as plt

def plot_metric_comparison(
    results_dict,
    include_models=None,
    include_offsets=None,
    log_y=False,
    log_x=False,
    metric="Rouge-L",
):
    # Disable TeX interpretation to allow % characters
    plt.rcParams["text.usetex"] = False
    """
    Plot comparison of models with metric scores across different repetitions
    
    Parameters
    ----------
    results_dict : dict
        Dictionary of Results objects with structure {model_name: Results_object}
    include_models : list, optional
        List of model names to include in plotting. If None, includes all models.
    include_offsets : list, optional
        List of offset values to include in plotting. If None, includes all offsets.
    log_y : bool, optional
        Whether to use log scale for y-axis
    log_x : bool, optional
        Whether to use log scale for x-axis
    metric : str, optional
        Metric to plot (e.g., "Rouge-L", "TTR-gen"). Default is "Rouge-L"
    """
    plt.figure(figsize=(12, 6))

    # Line styles and markers for different offsets and models
    styles = ["-", "--", ":", "-."]
    markers = ["o", "s", "^", "D", "v", "<", ">", "p"]

    # Filter models to plot
    models_to_plot = (
        list(results_dict.keys()) if include_models is None else include_models
    )

    # Check total number of offsets for simplified labels
    all_offsets = set()
    for model_name in models_to_plot:
        if model_name in results_dict:
            result_obj = results_dict[model_name]
            all_offsets.update(result_obj.offsets)

    # Count total unique offsets across all models after filtering
    if include_offsets is not None:
        filtered_offsets = set(include_offsets).intersection(all_offsets)
        single_offset = len(filtered_offsets) == 1
    else:
        single_offset = len(all_offsets) == 1

    for model_idx, model_name in enumerate(models_to_plot):
        if model_name not in results_dict:
            continue

        result_obj = results_dict[model_name]
        expr_name = result_obj.expr[0]

        # Get available offsets and filter if needed
        available_offsets = result_obj.offsets
        offsets_to_plot = (
            available_offsets
            if include_offsets is None
            else [o for o in include_offsets if o in available_offsets]
        )

        # Default prefix and suffix values (assuming they're consistent)
        prefix = result_obj.prefixes[0]
        suffix = result_obj.suffixes[0]

        for offset_idx, offset in enumerate(offsets_to_plot):
            # Get repetitions for this model
            repetitions = sorted(result_obj.repetitions)

            # Get metric scores for each repetition
            x_values = []
            metric_scores = []

            for rep in repetitions:
                try:
                    stats = result_obj.get_stats(
                        expr_name, rep, offset, prefix, suffix, metric
                    )
                    metric_scores.append(stats.mean)
                    x_values.append(rep)  # Keep original x values
                except KeyError:
                    # Skip this repetition if it doesn't exist for this model
                    continue

            # Select style and marker based on offset index and model index
            style_idx = offset_idx % len(styles)
            marker_idx = model_idx % len(markers)

            # Use short model name for display
            display_name = model_name
            if "llama3-1b-" in model_name:
                if "standard" in model_name:
                    display_name = "standard"
                elif "goldfish" in model_name:
                    # Extract the k value from the model name
                    if "k-283" in model_name:
                        display_name = "goldfish-drop-0.5" + chr(
                            37
                        )  # Use ASCII code for %
                    elif "k-54" in model_name:
                        display_name = "goldfish-drop-2.0" + chr(37)
                    elif "k-21" in model_name:
                        display_name = "goldfish-drop-5.0" + chr(37)
                    elif "k-10" in model_name:
                        display_name = "goldfish-drop-10.0" + chr(37)
                    elif "k-5" in model_name:
                        display_name = "goldfish-drop-20.0" + chr(37)
                    else:
                        # Extract k value
                        k_match = re.search(r"k-(\d+)", model_name)
                        if k_match:
                            display_name = f"goldfish-k{k_match.group(1)}"
                        else:
                            display_name = "goldfish"

            # Create label - don't show offset if there's only one
            if single_offset:
                label = f"{display_name}"
            else:
                label = f"{display_name} (offset={offset})"

            # Plot the data
            plt.plot(
                x_values,
                metric_scores,
                linestyle=styles[style_idx],
                marker=markers[marker_idx],
                label=label,
                alpha=0.7,
            )

    plt.xlabel("Repetitions")
    plt.ylabel(f"Average {metric} Score")

    # Set axis scales if requested
    if log_y:
        plt.yscale("log")
    if log_x:
        plt.xscale("log", base=2)  # Use base 2 for power of 2 scale

    # Improve legend appearance with light background
    plt.legend(
        bbox_to_anchor=(1.05, 1), loc="upper left", facecolor="white", edgecolor="black"
    )
    plt.grid(True)
    plt.tight_layout()

    plt.show()