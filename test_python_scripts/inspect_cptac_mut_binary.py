import pandas as pd
df_b = pd.read_csv(r"D:\data\CPTAC\Mutation_BCM_v1\BRCA_somatic_mutation_gene_level_binary.txt", sep="\t", nrows=4)
print("CPTAC Mut cols[:6]:", list(df_b.columns[:6]), "total cols:", len(df_b.columns))
print(df_b.iloc[:3, :5].to_string(index=False))
