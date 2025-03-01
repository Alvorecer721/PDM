import json
import glob
import os

def find_json_in_folder(folder_path):
    """
    Find JSON files in the specified folder and its immediate subdirectories.
    
    Args:
        folder_path (str): Path to the folder to search
        
    Returns:
        str or None: Path to the first JSON file found, or None if no JSON files exist
    """
    try:
        # Ensure folder_path exists
        if not os.path.isdir(folder_path):
            print(f"Warning: {folder_path} is not a valid directory")
            return None
            
        # First, search directly in the specified folder
        json_files = glob.glob(os.path.join(folder_path, "*.json"))
        if json_files:
            return json_files[0]
            
        # If no JSON files found in the main folder, search in subdirectories
        for subdir in os.listdir(folder_path):
            subdir_path = os.path.join(folder_path, subdir)
            if os.path.isdir(subdir_path):
                json_files = glob.glob(os.path.join(subdir_path, "*.json"))
                if json_files:
                    return json_files[0]
                    
        print(f"No JSON files found in {folder_path} or its subdirectories")
        return None
            
    except Exception as e:
        print(f"Error searching for JSON files: {str(e)}")
        return None


# Function to load results and stderr values
def load_results_with_stderr(expr_dir):
    json_path = find_json_in_folder(expr_dir)
    with open(json_path, 'r') as file:
        data = json.load(file)
    
    # Extract metrics and their standard errors
    metrics = {
        'ARC-Challenge': {
            'acc': data['results']['arc_challenge']['acc,none'],
            'acc_stderr': data['results']['arc_challenge']['acc_stderr,none'],
            'acc_norm': data['results']['arc_challenge']['acc_norm,none'],
            'acc_norm_stderr': data['results']['arc_challenge']['acc_norm_stderr,none'],
        },
        'ARC-Easy': {
            'acc': data['results']['arc_easy']['acc,none'],
            'acc_stderr': data['results']['arc_easy']['acc_stderr,none'],
            'acc_norm': data['results']['arc_easy']['acc_norm,none'],
            'acc_norm_stderr': data['results']['arc_easy']['acc_norm_stderr,none'],
        },
        'CommonsenseQA': {
            'acc': data['results']['commonsense_qa']['acc,none'],
            'acc_stderr': data['results']['commonsense_qa']['acc_stderr,none']
        },
        'HellaSwag': {
            'acc': data['results']['hellaswag']['acc,none'],
            'acc_stderr': data['results']['hellaswag']['acc_stderr,none'],
            'acc_norm': data['results']['hellaswag']['acc_norm,none'],
            'acc_norm_stderr': data['results']['hellaswag']['acc_norm_stderr,none']
        },
        'MMLU': {
            'acc': data['groups']['mmlu']['acc,none'],
            'acc_stderr': data['groups']['mmlu']['acc_stderr,none']
        },
        'PIQA': {
            'acc': data['results']['piqa']['acc,none'],
            'acc_stderr': data['results']['piqa']['acc_stderr,none'],
            'acc_norm': data['results']['piqa']['acc_norm,none'],
            'acc_norm_stderr': data['results']['piqa']['acc_norm_stderr,none']
        },
        'Winogrande': {
            'acc': data['results']['winogrande']['acc,none'],
            'acc_stderr': data['results']['winogrande']['acc_stderr,none']
        },
        "wikitext": {
            'word_perplexity': data['results']['wikitext']['word_perplexity,none'],
            'word_perplexity_stderr': data['results']['wikitext']['word_perplexity_stderr,none'],
        },
        "SQuADv2": {
            'best_exact': data['results']['squadv2']['best_exact,none']
        }
    }
    return metrics


def create_latex_table(model_dict, digits=2, caption="Benchmark Results", label="tab:benchmarks", font_size="tiny"):
    """
    Generates a compact LaTeX table with grouped columns and no standard deviations.
    Formats accuracy values as percentages (multiplied by 100).
    
    Args:
        model_dict: Dictionary mapping model names to dataset metrics
        digits: Number of decimal places to show (default: 2)
        caption: Table caption (default: "Benchmark Results") 
        label: Table label for referencing (default: "tab:benchmarks")
        font_size: LaTeX font size command (default: "tiny")
        
    Returns:
        str: LaTeX table code with proper formatting
    """
    # Define benchmarks and their grouped metrics
    grouped_benchmarks = [
        ('Wiki.', 'wikitext', ['word ppl'], False),  # (short_name, data_key, metrics, is_grouped)
        ('Hella.', 'HellaSwag', ['acc', 'acc norm'], True),
        ('ARC-c', 'ARC-Challenge', ['acc', 'acc norm'], True),
        ('PIQA', 'PIQA', ['acc'], False),
        ('Wino.', 'Winogrande', ['acc'], False),
        ('ARC-e', 'ARC-Easy', ['acc'], False),
        ('CSQA', 'CommonsenseQA', ['acc'], False),
        ('MMLU', 'MMLU', ['acc'], False),  # Fixed missing comma here
        ('SQuADv2', 'SQuADv2', ['best exact'], False)  # Already included in the list
    ]
    
    # Generate LaTeX with proper indentation and line breaks
    latex = []
    
    # Table environment
    latex.append("\\begin{table}[ht]")
    latex.append(f"    \\{font_size}")
    latex.append("    \\centering")
    latex.append(f"    \\caption{{{caption}}}")
    latex.append(f"    \\label{{{label}}}")
    
    # Determine tabular format
    # For each benchmark: if not grouped, use 1 column; if grouped, use N columns
    col_specs = ["l"]  # First column for model names
    
    # Build the tabular format string directly without join to avoid double separators
    tabular_format = "l"  # Start with the model name column
    
    for _, _, metrics, is_grouped in grouped_benchmarks:
        tabular_format += "|"  # Add a single separator between column groups
        
        if is_grouped:
            # Add multiple columns for grouped metrics
            tabular_format += "c" * len(metrics)
        else:
            # Add a single column
            tabular_format += "c"
    
    latex.append(f"    \\begin{{tabular}}{{{tabular_format}}}")
    latex.append("        \\toprule")
    
    # Create first header row with benchmark names and multicolumns
    header1 = ["\\multirow{2}{*}{Model}"]
    for benchmark_short, _, metrics, is_grouped in grouped_benchmarks:
        if is_grouped:
            # Add a multicolumn for grouped metrics
            header1.append(f"\\multicolumn{{{len(metrics)}}}{{c|}}{{{benchmark_short}}}")
        else:
            # Add a single column
            header1.append(benchmark_short)
    
    # Create second header row with metric types
    header2 = [""]  # Empty for the model column
    for _, _, metrics, _ in grouped_benchmarks:
        for metric in metrics:
            # Map the display names
            if metric == "word ppl":
                direction = "↓"
            elif metric == "best exact":  # Add direction for SQuADv2's best_exact metric
                direction = "↑"
            else:
                direction = "↑"
                
            # Format as "metric_name↑" or "metric_name↓"
            if metric == "word ppl":
                header2.append(f"ppl{direction}")
            elif metric == "acc norm":
                header2.append(f"norm{direction}")
            elif metric == "best exact":  # Special formatting for best_exact
                header2.append(f"exact{direction}")
            else:  # acc
                header2.append(f"acc{direction}")
    
    # Add headers with indentation
    latex.append("        " + " & ".join(header1) + " \\\\")
    latex.append("        " + " & ".join(header2) + " \\\\")
    latex.append("        \\midrule")
    
    # Add data rows
    for model_name, model_data in model_dict.items():
        row = [model_name]
        
        for _, benchmark_full, metrics, _ in grouped_benchmarks:
            for metric in metrics:
                # Map the display names back to the actual keys in the data
                if metric == "word ppl":
                    data_key = "word_perplexity"
                elif metric == "acc norm":
                    data_key = "acc_norm"
                elif metric == "best exact":
                    data_key = "best_exact"  # Use the correct key for SQuADv2
                else:
                    data_key = metric
                
                if benchmark_full in model_data and data_key in model_data[benchmark_full]:
                    value = model_data[benchmark_full][data_key]
                    
                    # Format the value (no standard deviation)
                    try:
                        float_value = float(value)
                        
                        # Format accuracy metrics as percentages (multiply by 100)
                        if data_key in ["acc", "acc_norm"]:  # Only multiply acc metrics, not best_exact
                            float_value *= 100  # Convert to percentage
                            
                        row.append(f"{float_value:.{digits}f}")
                    except (ValueError, TypeError):
                        # Value is not numeric (e.g., 'N/A')
                        row.append(str(value))
                else:
                    row.append("--")
        
        # Add data row with indentation
        latex.append("        " + " & ".join(row) + " \\\\")
    
    # Finish LaTeX table with proper indentation
    latex.append("        \\bottomrule")
    latex.append("    \\end{tabular}")
    latex.append("\\end{table}")
    
    # Join with actual line breaks (not escaped \n)
    return "\n".join(latex)