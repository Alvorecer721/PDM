from exact_match import CommonSubstringMatcher

if __name__ == "__main__":
    from datasets import load_dataset
    from time import time

    s1 = [1, 2, 4, 3, 5, 6, 7, 11, 9, 10, 12, 13, 15, 14, 16, 17, 18, 20, 19]
    s2 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16, 17, 18, 19, 20]

    # First run (includes compilation time)
    start = time()
    result = find_relaxed_matches(
        s1, 
        s2, 
        min_consecutive=1, 
        max_edits=0, 
    )
    print(f"First run (warmup with compilation): {time() - start:.2f}s")

    # Load the JSONL file
    dataset = load_dataset(
        "json",
        # data_files="/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Standard_GBS_120_EPOCH_75/step=2400-consumed=288000/rank0.jsonl", 
        # data_files="/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Standard_GBS_120_EPOCH_75/step=5400-consumed=648000/rank0.jsonl",  
        # data_files="/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Goldfish_K_10_H_13_GBS_120_EPOCH_83/step=1800-consumed=216000/rank0.jsonl",
        data_files="/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Goldfish_K_5_H_13_GBS_120_EPOCH_93/step=2400-consumed=288000/rank0.jsonl",
        # data_files='/mloscratch/homes/yixuan/PDM/inference/llama_1.5B_Goldfish_K_21_H_13_GBS_120_EPOCH_79/step=4200-consumed=504000/rank0.jsonl', 
        split="train",
    )

    num_tokens = 500
    s1 = dataset[0]["true_suffix"][:num_tokens]
    s2 = dataset[0]["generated_suffix"][:num_tokens]

    # Second run (actual runtime)
    start = time()
    result = find_relaxed_matches(s1, s2, min_consecutive=1, max_edits=0)
    end = time()
    print(f"Second run (actual runtime): {end - start:.2f}s")

    # Print results
    for match in result:
        print(
            f"Match(length={match.length}, edits={match.edits}, "
            f"s1[{match.start_pos1}:{match.end_pos1}]={match.values1}, "
            f"s2[{match.start_pos2}:{match.end_pos2}]={match.values2})"
        )
