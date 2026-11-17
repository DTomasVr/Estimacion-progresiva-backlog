"""Unify TAWOS + Public Jira into one node table, one parent->child edge table
and one project table, then profile what hierarchy is actually available.

Edges:
  * Jira  subtask : fields.parent.key          -> key
  * Jira  epic    : Epic Link customfield       -> key
  * TAWOS subtask : last 'Parent'/'Parent Issue' change per issue (Change_Log)
  * TAWOS epic    : last 'Epic Link' change per issue, plus 'Epic Child' pairs
                    whose child never had an Epic Link change logged

Overlap: a TAWOS project whose repository is also a Public Jira instance and
whose key exists there is taken from Jira (later snapshot, native parent
fields); TAWOS only lends it its project Description. CLI/DAEMON in TAWOS are
Appcelerator's, not Apache's, so the mapping goes through the repository.

No text cleaning here -- that is preprocessing.
"""
import glob
import time

import pandas as pd

PBE = r"C:\Users\User\Documents\Programacao\PBE-Dataset\data\interim"
BASE = r"F:\Software Task generation Model"
REPO_A_JIRA = {"Spring": "Spring", "Sonatype": "Sonatype", "Apache": "Apache",
               "Atlassian": "Jira", "Hyperledger": "Hyperledger", "MongoDB": "MongoDB"}
t0 = time.time()


def fase(msg):
    print(f"[{time.time()-t0:6.1f}s] {msg}", flush=True)


# ---------------------------------------------------------------- Public Jira
cols = ["source", "key", "project_key", "project_name", "project_category",
        "project_category_desc", "issuetype", "is_subtask", "status", "resolution",
        "summary", "description", "created", "parent_key", "epic_link", "epic_name"]
def leer_parte(f):
    # a part lacks a column when none of its 100k rows had that field set
    import pyarrow.parquet as pq
    hay = set(pq.read_schema(f).names)
    return pd.read_parquet(f, columns=[c for c in cols if c in hay]).reindex(columns=cols)


J = pd.concat([leer_parte(f)
               for f in sorted(glob.glob(f"{BASE}/data/interim/jira_full/*/*.parquet"))],
              ignore_index=True)
J["dataset"] = "jira"
J["uid"] = J.source + ":" + J.key
fase(f"Jira cargado {len(J):,}")

ej = []
m = J.parent_key.notna()
ej.append(pd.DataFrame({"parent_uid": J.source[m] + ":" + J.parent_key[m],
                        "child_uid": J.uid[m], "relation": "subtask"}))
m = J.epic_link.notna()
ej.append(pd.DataFrame({"parent_uid": J.source[m] + ":" + J.epic_link[m],
                        "child_uid": J.uid[m], "relation": "epic"}))
EJ = pd.concat(ej, ignore_index=True)
EJ["dataset"] = "jira"

# ---------------------------------------------------------------------- TAWOS
P = pd.read_parquet(f"{PBE}/Project.parquet")
R = pd.read_parquet(f"{PBE}/Repository.parquet")[["ID", "Name"]].rename(
    columns={"ID": "Repository_ID", "Name": "repo"})
P = P.merge(R, on="Repository_ID", how="left")
P["jira_source"] = P.repo.map(REPO_A_JIRA)
jira_keys = set(zip(J.source, J.project_key))
P["overlap"] = [(s, k) in jira_keys for s, k in zip(P.jira_source, P.Project_Key)]
fase(f"TAWOS proyectos solapados con Jira: {int(P.overlap.sum())} de {len(P)}")

T = pd.read_parquet(f"{PBE}/Issue.parquet",
                    columns=["ID", "Issue_Key", "Title", "Description_Text", "Type",
                             "Status", "Resolution", "Creation_Date", "Project_ID"])
T = T.merge(P[["ID", "Project_Key", "Name", "repo", "overlap"]].rename(columns={"ID": "Project_ID"}),
            on="Project_ID", how="left")
T["uid"] = "TAWOS-" + T.repo + ":" + T.Issue_Key

H = pd.read_parquet(f"{BASE}/data/interim/tawos_changelog_hier.parquet")
H["Issue_ID"] = H.Issue_ID.astype("int64")
H["ID"] = H.ID.astype("int64")
H = H.merge(T[["ID", "uid", "repo"]].rename(columns={"ID": "Issue_ID"}), on="Issue_ID", how="inner")
H = H.sort_values(["Creation_Date", "ID"])
H["grupo"] = H.Field.map({"Epic Link": "epic", "Parent": "subtask", "Parent Issue": "subtask"})

et = []
ult = H[H.grupo.notna()].groupby(["uid", "grupo"]).tail(1)
ult = ult[ult.To_String.notna()]
et.append(pd.DataFrame({"parent_uid": "TAWOS-" + ult.repo + ":" + ult.To_String,
                        "child_uid": ult.uid, "relation": ult.grupo}))
ec = H[H.Field == "Epic Child"].copy()
ec["child_key"] = ec.To_String.fillna(ec.From_String)
ec = ec.groupby(["uid", "child_key"]).tail(1)
ec = ec[ec.To_String.notna()]
ec_df = pd.DataFrame({"parent_uid": ec.uid, "child_uid": "TAWOS-" + ec.repo + ":" + ec.child_key,
                      "relation": "epic"})
con_epic_link = set(H.loc[H.grupo == "epic", "uid"])
et.append(ec_df[~ec_df.child_uid.isin(con_epic_link)])
ET = pd.concat(et, ignore_index=True).drop_duplicates()
ET["dataset"] = "tawos"
# drop TAWOS projects that Jira already covers
T_keep = T[~T.overlap.astype(bool)]
ET = ET[ET.child_uid.isin(set(T_keep.uid)) & ET.parent_uid.isin(set(T_keep.uid))]
fase(f"TAWOS aristas {len(ET):,} (tras quitar solapados)")

# ---------------------------------------------------------------- unify nodes
Tn = pd.DataFrame({"uid": T_keep.uid, "dataset": "tawos", "source": "TAWOS-" + T_keep.repo,
                   "key": T_keep.Issue_Key, "project_key": T_keep.Project_Key,
                   "project_name": T_keep.Name, "issuetype": T_keep.Type,
                   "is_subtask": T_keep.Type.eq("Sub-task"), "status": T_keep.Status,
                   "resolution": T_keep.Resolution, "summary": T_keep.Title,
                   "description": T_keep.Description_Text,
                   "created": T_keep.Creation_Date.astype(str)})
Jn = J.drop(columns=["parent_key", "epic_link"])
N = pd.concat([Jn, Tn], ignore_index=True)
# the mongodump repeats 59 issues verbatim at batch boundaries
N = N.drop_duplicates("uid", keep="first").reset_index(drop=True)
N["project_uid"] = N.source + ":" + N.project_key
E = pd.concat([EJ, ET], ignore_index=True)
nodos = set(N.uid)
E["parent_existe"] = E.parent_uid.isin(nodos)
fase(f"nodos {len(N):,}  aristas {len(E):,}  con padre presente {E.parent_existe.mean():.1%}")
E = E[E.parent_existe].drop(columns="parent_existe").drop_duplicates(["parent_uid", "child_uid", "relation"])

# ------------------------------------------------------------------ projects
PR = N.groupby("project_uid").agg(dataset=("dataset", "first"), source=("source", "first"),
                                  project_key=("project_key", "first"),
                                  project_name=("project_name", "first"),
                                  category=("project_category", "first"),
                                  category_desc=("project_category_desc", "first"),
                                  n_issues=("uid", "size")).reset_index()
pdesc = P.assign(project_uid=lambda d: d.jira_source.where(d.overlap, "TAWOS-" + d.repo) + ":" + d.Project_Key)
PR = PR.merge(pdesc[["project_uid", "Description", "URL"]].rename(columns={"Description": "description"}),
              on="project_uid", how="left")
fase(f"proyectos {len(PR):,}; con descripcion {PR.description.notna().sum()}")

N.to_parquet(f"{BASE}/data/interim/nodes.parquet", index=False)
E.to_parquet(f"{BASE}/data/interim/edges.parquet", index=False)
PR.to_parquet(f"{BASE}/data/interim/projects.parquet", index=False)
fase("escrito nodes/edges/projects")

# ------------------------------------------------------------------- profile
tipo = N.set_index("uid").issuetype
E["parent_type"] = E.parent_uid.map(tipo)
E["child_type"] = E.child_uid.map(tipo)
E["src"] = E.parent_uid.str.split(":").str[0]
filas = []
for (src, rel), g in E.groupby(["src", "relation"]):
    k = g.groupby("parent_uid").size()
    filas.append(dict(source=src, relation=rel, padres=len(k), aristas=len(g),
                      hijos_mediana=k.median(), hijos_p90=k.quantile(.9),
                      padres_2a15=int(k.between(2, 15).sum())))
prof = pd.DataFrame(filas).sort_values(["relation", "aristas"], ascending=[True, False])
print(prof.to_string(index=False))
prof.to_csv(f"{BASE}/results/03_hierarchy_profile.csv", index=False)

for rel in ["epic", "subtask"]:
    g = E[E.relation == rel]
    print(f"\n{rel}: tipos de padre\n", g.parent_type.value_counts().head(10).to_string())
    print(f"{rel}: tipos de hijo\n", g.child_type.value_counts().head(10).to_string())

# 3-level chains: epic -> child -> subtasks
hijos_epic = set(E.loc[E.relation == "epic", "child_uid"])
padres_sub = set(E.loc[E.relation == "subtask", "parent_uid"])
print(f"\ncadenas epic->historia->subtareas: {len(hijos_epic & padres_sub):,} historias")
print(f"total {time.time()-t0:.0f}s")
