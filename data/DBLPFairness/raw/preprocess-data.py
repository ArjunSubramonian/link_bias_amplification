import pandas as pd
import numpy as np

df = pd.read_csv("dblp_new_version.csv")

papers = {}
for index, row in df.iterrows():
    try:
        p_year = int(row['p_year'])
        if p_year < 2010:
            continue
        
        tid, aorgc, refs, fos = row['tid'], row['aorgc'].lower(), row["references"].lower(), row["fos"].lower()
        team_size, mean_collaborator, gini_collaborator, mean_productivity, gini_productivity = \
            row['team_size'], row['mean_collaborator'], row["gini_collaborator"], \
            row["mean_productivity"], row["gini_productivity"]
    except:
        continue
    
    aorgc_list = aorgc.strip('[]').replace(" ", "").replace("'", "").split(",")
    refs_list = refs.strip('[]').replace(" ", "").replace("'", "").split(",")
    fos_list = fos.strip('[]').replace(" ", "").replace("'", "").split(",")
   
    majc = max(aorgc_list, key=aorgc_list.count)
    if len(majc) == 0:
        continue
    if majc == "unitedstates":
        majc = 0
    elif majc == "china":
        majc = 1
    else:
        continue
        
    if "programminglanguage" not in fos_list and "database" not in fos_list:
        continue
    elif "programminglanguage" in fos_list and "database" in fos_list:
        continue
    elif "programminglanguage" in fos_list:
        majfos = 0
    else:
        majfos = 1

    # print(tid, majc, majfos, team_size, mean_collaborator, gini_collaborator, mean_productivity, gini_productivity)
    papers[tid] = [majc, majfos, team_size, mean_collaborator, gini_collaborator, mean_productivity, gini_productivity, [int(ref) for ref in refs_list]]
    
papers_df = pd.DataFrame.from_dict(papers, orient='index', columns=['country', 'field', 'team_size', 'mean_collaborator', 'gini_collaborator', 'mean_productivity', 'gini_productivity', 'references'])
index_to_node = dict(zip(papers_df.index.values, list(range(len(papers_df.index.values)))))

row = []
col = []

for index, refs in papers_df["references"].items():
    src = index_to_node[index]
    tgt = [index_to_node[nid] for nid in refs if nid in index_to_node]
    
    row.extend([src for _ in range(len(tgt))])
    col.extend(tgt)

edge_index = np.array([row, col])
data={
    'features' : papers_df[papers_df.columns.difference(['field', 'references'])].values,
    'target' : papers_df['field'].values,
    'edges' : edge_index
}
np.savez('dblp_fairness.npz', \
        features=papers_df[papers_df.columns.difference(['field', 'references'])].values, \
        target=papers_df['field'].values, \
        edges=edge_index
       )
