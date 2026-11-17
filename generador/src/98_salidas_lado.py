import json,sys,ast,pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
A="C:/Users/User/Documents/Programacao/PBE-Dataset/results/banco_generacion/"
L=[json.loads(l) for l in open("F:/Software Task generation Model/results/20_banco.jsonl",encoding="utf-8")]
L+=[json.loads(l) for f in ("pbe_actual.jsonl","groq__openai_gpt-oss-120b.jsonl") for l in open(A+f,encoding="utf-8")]
caso=sys.argv[1]
d=pd.read_csv(A+"por_proyecto.csv"); d=d[d.caso==caso].set_index("sistema")
for r in L:
    if r["caso"]!=caso or r["condicion"] in ("ejemplos","-"): continue
    H=ast.literal_eval(r["historias"]) if isinstance(r["historias"],str) else r["historias"]
    T=ast.literal_eval(r["tareas"]) if isinstance(r["tareas"],str) else r["tareas"]
    print("==",r["sistema"],r["condicion"],"n_tareas",r["n_tareas"],"seg",r["segundos"],"sonda",r.get("sonda"))
    for h,t in list(zip(H,T))[:2]: print("  H:",h); [print("     -",x) for x in t[:4]]; print("     (",len(t),"tareas)")
print(d[["sF1","n_tareas","segundos"]].round(3).to_string())
