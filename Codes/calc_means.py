import pandas as pd, os
models = ['deepseek', 'gpt4o_mini', 'grok', 'qwen72b']
for m in models:
    path = f'results/study1_sar_{m}.csv'
    if os.path.exists(path):
        df = pd.read_csv(path)
        mean_gs = df['gen_sar'].dropna().mean()
        print(f'{m}: gen_sar mean = {mean_gs:.4f}')
