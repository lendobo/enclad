# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

spectral_data_all = '/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/ENCLAD/phase_1_code/data/41597_2015_BFsdata201522_MOESM56_ESM.xls'
cms2_path = '/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/ENCLAD/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS2.csv'
cms4_path = '/home/celeroid/Documents/CLS_MSc/Thesis/EcoCancer/ENCLAD/phase_1_code/data/Synapse/TCGA/Proteomics_CMS_groups/TCGACRC_proteomics_CMS4.csv'

spectral = pd.read_excel(spectral_data_all, index_col=0)
cms2 = pd.read_csv(cms2_path, index_col=0)
cms4 = pd.read_csv(cms4_path, index_col=0)

spectral.head()
# %%
# Apply row-wise mean
spectral['mean'] = spectral.mean(axis=1)
cms2['mean'] = cms2.mean(axis=1)
cms4['mean'] = cms4.mean(axis=1)

# Apply row-wise SD
spectral['sd'] = spectral.std(axis=1)
cms2['sd'] = cms2.std(axis=1)
cms4['sd'] = cms4.std(axis=1)

# Plot meanvs SD for each CMS group, in red and blue
sns.scatterplot(data=spectral, x='mean', y='sd', color='black', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.show()

# %%
sns.scatterplot(data=cms2, x='mean', y='sd', color='red', alpha=0.5)
sns.scatterplot(data=cms4, x='mean', y='sd', color='blue', alpha=0.3)
# plt.xscale('log')
# plt.yscale('log')
plt.show()
# %%
