import pandas as pd, glob
study3_dfs = [pd.read_csv(f) for f in glob.glob("results/study3_gss_*.csv")]
if study3_dfs:
    s3 = pd.concat(study3_dfs, ignore_index=True)
    s3["gss"] = s3["gss_mean"]
    macro = s3.groupby(["model_name","category"])["gss"].mean().reset_index().groupby("model_name")["gss"].mean().sort_values()
    print("MACRO GSS ranks:")
    for i,(m,v) in enumerate(macro.items(), 1):
        print(f"  rank {i}: {m} = {v:.4f}")
else:
    print("No Study 3 files found.")

study1_dfs = [pd.read_csv(f) for f in glob.glob("results/study1_sar_*.csv")]
if study1_dfs:
    s1 = pd.concat(study1_dfs, ignore_index=True)
    API = {"deepseek","gpt4o_mini","grok","qwen72b"}
    rep = {}
    for model, grp in s1.groupby("model_name"):
        if model in API:
            v = grp["gen_sar"].dropna().mean()
            if pd.notna(v): rep[model] = v
        else:
            v = grp[grp["prefers_stereo"]>=0]["prefers_stereo"].mean()
            if pd.notna(v): rep[model] = v
    rep_sorted = sorted(rep, key=lambda m: rep[m])
    print("REP ranks:")
    for i,m in enumerate(rep_sorted, 1):
        print(f"  rank {i}: {m} = {rep[m]:.4f}")
else:
    print("No Study 1 files found.")
