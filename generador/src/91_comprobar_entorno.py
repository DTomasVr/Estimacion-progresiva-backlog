"""Comprueba que todo lo que el pipeline necesita esta en su sitio, sin cargar modelos.

  python src/91_comprobar_entorno.py

Sale con codigo 1 si falta algo, para poder encadenarlo en un script.
"""
import os
import sys

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NECESARIO = [
    ("modelo de arriba", "models/mt5-small-pruned_o2o_pbe_v1_o2ofmt_pbe_v1_v9bt_pbe_v1"),
    ("modelo de abajo", "models/mt5-small-pruned_hist_pbe_v1_v5_pbe_v1"),
    # el conjunto por omision (PBE_CONJUNTO=dos_modelos) carga TAMBIEN este;
    # sin el, la comprobacion pasaba y el pipeline reventaba al cargar
    ("modelo de abajo, 2.o del conjunto", "models/mt5-small-pruned_hist_pbe_v1_bs8_lr9.2e-4"),
    ("SBERT propio", "models/sbert_pbe_v1"),
    ("reordenador de beneficios", "models/reordenador_beneficio.joblib"),
    ("embeddings del corpus", "data/processed/pipeline_emb.npz"),
    ("tabla de vecinos de arriba", "data/processed/s2s_o2o_train.parquet"),
    ("tabla de vecinos de abajo", "data/processed/s2s_hist_train.parquet"),
    ("lexico de roles", "data/processed/roles_lexico.json"),
    ("clases de funcion", "data/processed/clases_funcion.json"),
    ("matriz rol x verbo", "data/processed/matriz_rol_verbo.json"),
    ("lexico de Jira estricto", "data/processed/lexico_jira_estricto.json"),
    ("taxonomia de 40 sectores", "data/synth/descripciones_v3.json"),
    ("proyectos reales para evaluar", "data/reales/test_real.json"),
]
OPCIONAL = [
    ("cache de los traductores Marian", "models/hf_cache"),
]


def main():
    print(f"proyecto: {BASE}\n")
    falta = []
    for etq, rel in NECESARIO:
        hay = os.path.exists(f"{BASE}/{rel}")
        print(f"  [{'ok' if hay else 'FALTA'}] {etq:<30} {rel}")
        if not hay:
            falta.append(rel)
    for etq, rel in OPCIONAL:
        hay = os.path.exists(f"{BASE}/{rel}")
        print(f"  [{'ok' if hay else '--'}] {etq:<30} {rel}"
              + ("" if hay else "   (se descarga solo la primera vez)"))
    print()
    try:
        import torch
        print(f"  torch {torch.__version__}   CUDA disponible: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            libre, total = torch.cuda.mem_get_info()
            print(f"  GPU: {torch.cuda.get_device_name(0)}  "
                  f"{libre / 2**30:.1f} GB libres de {total / 2**30:.1f}")
    except Exception as e:
        falta.append("torch")
        print(f"  [FALTA] torch: {e}")
    for m in ("transformers", "sentence_transformers", "pandas", "joblib",
              "sentencepiece", "sklearn", "numpy"):
        try:
            __import__(m)
        except Exception:
            falta.append(m)
            print(f"  [FALTA] {m}")
    if falta:
        print(f"\nFALTAN {len(falta)}: " + ", ".join(falta))
        return 1
    print("\nTodo en su sitio. Prueba de humo:")
    print('  python src/19_pipeline.py --traducir "Una app para reservar canchas '
          'deportivas donde los socios ven la disponibilidad y el administrador '
          'gestiona los horarios."')
    return 0


if __name__ == "__main__":
    sys.exit(main())
