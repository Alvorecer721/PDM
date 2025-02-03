import os
from difflib import SequenceMatcher
from datetime import datetime
from typing import List, Tuple
from ..verbatim_eval.my_rouge import _compute_dp_matrix_2d, compute_rouge_l_2d, find_contributing_tokens
import numpy as np


def create_rouge_viz(true_tokens, model_tokens, contributing_pos1, contributing_pos2, tokenizer):
    true_html = []
    model_html = []
    
    for i, token in enumerate(true_tokens):
        token_text = tokenizer.decode([token]).replace('<', '&lt;').replace('>', '&gt;')
        if i in contributing_pos1:
            true_html.append(f'<span class="contributing">{token_text}</span>')
        else:
            true_html.append(f'<span class="non-contributing">{token_text}</span>')
            
    for i, token in enumerate(model_tokens):
        token_text = tokenizer.decode([token]).replace('<', '&lt;').replace('>', '&gt;')
        if i in contributing_pos2:
            model_html.append(f'<span class="contributing">{token_text}</span>')
        else:
            model_html.append(f'<span class="non-contributing">{token_text}</span>')
            
    return ''.join(true_html), ''.join(model_html)


def log_model_generations(
    model_tokens,
    true_tokens,
    tokenizer,
    rep_count: int,
    seq_idx: int,
    output_dir: str
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"generation_log_rep{rep_count}_seq{seq_idx}_{timestamp}.html"
    filepath = os.path.join(output_dir, filename)

    true_text = tokenizer.decode(true_tokens)
    model_text = tokenizer.decode(model_tokens)
    
    # Diff visualization
    matcher = SequenceMatcher(None, true_tokens, model_tokens)
    true_diff_html = []
    model_diff_html = []
    
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == 'equal':
            true_span = tokenizer.decode(true_tokens[i1:i2]).replace('<', '&lt;').replace('>', '&gt;')
            model_span = tokenizer.decode(model_tokens[j1:j2]).replace('<', '&lt;').replace('>', '&gt;')
            true_diff_html.append(f'<span class="equal">{true_span}</span>')
            model_diff_html.append(f'<span class="equal">{model_span}</span>')
        elif tag == 'delete':
            true_span = tokenizer.decode(true_tokens[i1:i2]).replace('<', '&lt;').replace('>', '&gt;')
            true_diff_html.append(f'<span class="delete">{true_span}</span>')
        elif tag == 'insert':
            model_span = tokenizer.decode(model_tokens[j1:j2]).replace('<', '&lt;').replace('>', '&gt;')
            model_diff_html.append(f'<span class="insert">{model_span}</span>')
        elif tag == 'replace':
            true_span = tokenizer.decode(true_tokens[i1:i2]).replace('<', '&lt;').replace('>', '&gt;')
            model_span = tokenizer.decode(model_tokens[j1:j2]).replace('<', '&lt;').replace('>', '&gt;')
            true_diff_html.append(f'<span class="delete">{true_span}</span>')
            model_diff_html.append(f'<span class="insert">{model_span}</span>')

    # Rouge-L visualization
    model_tokens = model_tokens.cpu()
    dp = _compute_dp_matrix_2d(np.array(true_tokens), np.array(model_tokens))
    rouge_score = compute_rouge_l_2d(dp)
    lcs_length, contributing_tokens, pos1, pos2 = find_contributing_tokens(
        np.array(true_tokens), np.array(model_tokens))
    true_rouge_html, model_rouge_html = create_rouge_viz(
        true_tokens, model_tokens, pos1, pos2, tokenizer)


    css = """
.container {
    display: grid;
    grid-template-columns: 50px 1fr 50px 1fr;
    font-family: monospace;
    font-size: 14px;
    line-height: 1.5;
    margin-bottom: 20px;
}
.header {
    padding: 8px;
    background: #4a4a4a;
    color: white;
    font-weight: bold;
    text-align: center;
}
.line-numbers {
    padding: 0 8px;
    text-align: right;
    background: #e0e0e0;
    color: #666;
    white-space: pre;
    align-self: stretch;
}
.content {
    padding: 0 8px;
    white-space: pre;
    overflow-x: auto;
    background: white;
    align-self: stretch;
}
.equal { color: #333; }
.delete { background: #ffeef0; }
.insert { background: #e6ffed; }
.contributing { background: #fff3cd; }
.non-contributing { color: #999; }
.tabs {
    display: flex;
    margin-bottom: 1rem;
}
.tab {
    padding: 8px 16px;
    cursor: pointer;
    border: 1px solid #ccc;
    background: #f8f9fa;
    margin-right: 4px;
}
.tab.active {
    background: #fff;
    border-bottom: none;
}
.tab-content {
    display: none;
}
.tab-content.active {
    display: block;
}
.score {
    font-size: 16px;
    margin: 1rem 0;
    padding: 8px;
    background: #e9ecef;
    border-radius: 4px;
}
"""

    true_lines = true_text.count('\n') + 1
    model_lines = model_text.count('\n') + 1
    max_lines = max(true_lines, model_lines)

    html_template = """<!DOCTYPE html>
<html>
<head>
    <style>{css}</style>
    <script>
        function switchTab(tabName) {{
            document.querySelectorAll('.tab-content').forEach(content => {{
                content.classList.remove('active');
            }});
            document.querySelectorAll('.tab').forEach(tab => {{
                tab.classList.remove('active');
            }});
            document.getElementById(tabName).classList.add('active');
            document.getElementById(tabName + '-tab').classList.add('active');
        }}
    </script>
</head>
<body>
    <h2>Generation Log - Repetition {rep_count}, Sequence {seq_idx}</h2>
    <div class="tabs">
        <button class="tab active" id="diff-tab" onclick="switchTab('diff')">Diff View</button>
        <button class="tab" id="rouge-tab" onclick="switchTab('rouge')">Rouge-L View</button>
    </div>
    <div id="diff" class="tab-content active">
        <div class="container">
            <div class="header">Line</div>
            <div class="header">True Sequence</div>
            <div class="header">Line</div>
            <div class="header">Model Generation</div>
            <div class="line-numbers">{true_lines}</div>
            <div class="content">{true_diff_content}</div>
            <div class="line-numbers">{model_lines}</div>
            <div class="content">{model_diff_content}</div>
        </div>
    </div>
    <div id="rouge" class="tab-content">
        <div class="score">Rouge-L Score: {rouge_score:.4f}</div>
        <div class="container">
            <div class="header">Line</div>
            <div class="header">True Sequence</div>
            <div class="header">Line</div>
            <div class="header">Model Generation</div>
            <div class="line-numbers">{true_lines}</div>
            <div class="content">{true_rouge_content}</div>
            <div class="line-numbers">{model_lines}</div>
            <div class="content">{model_rouge_content}</div>
        </div>
    </div>
</body>
</html>"""

    html_content = html_template.format(
        css=css,
        rep_count=rep_count,
        seq_idx=seq_idx,
        true_lines='\n'.join(str(i) for i in range(1, max_lines + 1)),
        model_lines='\n'.join(str(i) for i in range(1, max_lines + 1)),
        true_diff_content=''.join(true_diff_html),
        model_diff_content=''.join(model_diff_html),
        true_rouge_content=true_rouge_html,
        model_rouge_content=model_rouge_html,
        rouge_score=rouge_score
    )

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)


if __name__ == "__main__":
    from transformers import AutoTokenizer, AutoModelForCausalLM

    model_path = "/iopsstor/scratch/cscs/xyixuan/experiment/llama_1.5B_Sparse_Gutenberg_K_50_H_13_GBS_60/results/HF"

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",  # Automatically handle device placement
        torch_dtype="auto"   # Automatically choose dtype
    )

    tokenizer = AutoTokenizer.from_pretrained('meta-llama/Meta-Llama-3-8B')

    # Example usage
    model_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    true_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]
    tokenizer = None
    rep_count = 1
    seq_idx = 0
    output_dir = "/capstor/users/cscs/xyixuan/PDM/results"
    
    log_model_generations(model_tokens, true_tokens, tokenizer, rep_count, seq_idx, output_dir)