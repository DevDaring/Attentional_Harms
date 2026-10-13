import pandas as pd, os, glob

models = ['qwen', 'deepseek', 'gemma3', 'grok', 'llama', 'gpt4o_mini', 'qwen72b']
model_labels = {'qwen':'Qwen2.5-1.5B','deepseek':'DeepSeek V3','gemma3':'Gemma-2-2B','grok':'Grok-3 Mini','llama':'Llama-3.1-8B','gpt4o_mini':'GPT-4o Mini','qwen72b':'Qwen2.5-72B'}
categories = ['caste','female','gender','male','race','religion']

# Collect all data
dfs = []
for m in models:
    fp = f'results/study4_multiturn_{m}.csv'
    if os.path.exists(fp):
        df = pd.read_csv(fp)
        dfs.append(df)
if dfs:
    all_df = pd.concat(dfs)

    # Per-model, per-category, per-depth means
    depths = ['gss_d0','gss_d2','gss_d4','gss_d6']
    # Filter depths that actually exist in the dataframe to avoid errors
    depths = [d for d in depths if d in all_df.columns]
    result = all_df.groupby(['model_name','category'])[depths].mean()
    print(result.to_string())
else:
    print('No data files found.')
