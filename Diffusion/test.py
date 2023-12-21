# %%
import pandas as pd

pathway_df = pd.read_csv('data/Pathway_Enrichment_Info.csv')

pathway_df = pathway_df.drop_duplicates(subset='description', keep='first')

num_duplicates = pathway_df['description'].duplicated().sum()
print("Number of duplicate descriptions:", num_duplicates)

# Filter the dataframe to include only pathways with '# genes' between 1 and 25
filtered_pathway_df = pathway_df[(pathway_df['# genes'] >= 10) & (pathway_df['# genes'] <= 25)]
# only keep first X
filtered_pathway_df = filtered_pathway_df.head(80)

# Extract the descriptions of these pathways
filtered_pathways = filtered_pathway_df['description'].tolist()

# Display the number of pathways and first few for inspection
num_filtered_pathways = len(filtered_pathways)
filtered_pathways[:10], num_filtered_pathways  # Displaying first 10 for brevity




