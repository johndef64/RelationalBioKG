# TODO — prossima sessione sul server

Obiettivo per la tesi: **R-GCN e CompGCN battono una baseline DistMult senza GNN**, allenata con lo
stesso protocollo, sugli stessi dati e sullo stesso split? Tutto il resto (ablation, repurposing)
viene dopo e usa gli stessi modelli.

Tutti i comandi si lanciano dalla root del repo. Gli script `experiments/*.sh` attivano da soli
l'env `gnn` e fissano `PYTHONHASHSEED=0` (senza, run identici davano risultati diversi).
Contesto e motivazioni: `docs/piano_consolidamento_v2.md`.

---

## 0. Preparazione (una volta)

```bash
git pull                                   # dopo il tuo commit su main
conda activate gnn && python -c "import torch; print(torch.cuda.is_available())"
wandb login                                # serve per l'HPO
ls dataset/PKT_subgraphs/                  # pkt_taskA_dti.tsv.zip, pkt_taskB_treats.tsv.zip, ablation/
```

### I risultati vecchi (HPO e ablation v1): NON cancellarli, archiviali

Non servono più per le tabelle nuove, ma non vanno buttati: documentano da dove si è partiti e
alcune config servono ancora.

| Cosa | Dove | Azione |
|---|---|---|
| Log dell'ablation parziale v1 e degli altri run v1 | server: `experiments/logs/*.log` | **spostali** in `experiments/logs/archive_v1/` (i nuovi log v2 finiscono in `experiments/logs/v2/`, ma così il resume dell'ablation non li confonde) |
| Modelli v1 | server: `models/` | cancellabili se serve spazio (non vengono riusati); altrimenti spostali in `models_archive_v1/` |
| Sweep W&B v1 | progetti `RelationalPKT-DTI-*`, `RelationalPKT-TREATS-*` | **lasciali**: l'HPO v2 scrive in progetti nuovi `...-v2-...` |
| Report HPO v1 | `experiments/hpo_best/`, `experiments/hpo_report/`, `docs/report_HPO_*.md` | **lasciali** (i file v2 hanno il suffisso `-v2`, niente sovrascritture) |
| Config tunate v1 | `src/models_params.json` → `PKT-DTI-best`, `PKT-TREATS-best` | **lasciale**: le usa E0 e fanno da fallback |

```bash
mkdir -p experiments/logs/archive_v1 && mv experiments/logs/*.log experiments/logs/archive_v1/ 2>/dev/null
```

---

## 1. E0 — confronto rapido vecchio vs nuovo protocollo (consigliato, poche ore)

Serve a una cosa sola: verificare sui dati veri che il protocollo v2 non peggiori rispetto a v1, e
avere subito un primo confronto con popolarità e DistMult. Usa le config già tunate.

```bash
TASKS=DTI CMP_MODELS=rgcn bash experiments/e0_protocol_compare.sh pair
python experiments/protocol_compare_summary.py          # -> experiments/protocol_compare_summary.md
```

Facoltativo, se c'è tempo (dice quale modifica ha l'effetto maggiore, utile per la sezione metodi):
`bash experiments/e0_protocol_compare.sh` (scala completa, entrambi i task, entrambe le GNN).

**Check prima di andare avanti:** in `protocol_compare_summary.md` la riga `v2` non deve avere M
nettamente peggiore di `+ fixed split`. Se succede, fermati e portami la tabella.

---

## 2. HPO v2 — tuning equo di R-GCN, CompGCN **e DistMult** (il passo più lungo)

```bash
bash experiments/e2_hpo_tandem.sh
```
- entrambi i task, 3 modelli, 30 trial ciascuno (`PKT_HPO_RUNS=20` per accorciare);
- progetti W&B: `RelationalPKT-DTI-v2-{rgcn,compgcn,distmult}` e `RelationalPKT-TREATS-v2-...`;
- alla fine scrive da solo le config migliori in `src/models_params.json` come
  `PKT-DTI-best-v2` e `PKT-TREATS-best-v2`.

Se si interrompe: `bash experiments/resume_hpo.sh A` (oppure `B`), poi di nuovo l'estrazione:
```bash
python experiments/get_best_hpo_config.py --task DTI    --suffix -v2 --write
python experiments/get_best_hpo_config.py --task TREATS --suffix -v2 --write
```
**Check:** `python -c "import json;d=json.load(open('src/models_params.json'));print({k:list(v) for k,v in d.items() if k.endswith('-v2')})"`
deve mostrare `rgcn`, `compgcn`, `distmult` per entrambi i task.

Nota memoria: CompGCN su TREATS con grafo completo è il caso più pesante; i trial in OOM vengono
registrati come saltati e lo sweep continua.

---

## 3. E1 — la tabella per la tesi: DistMult vs R-GCN vs CompGCN

```bash
PROTOCOL=v2 bash experiments/e1_main_training.sh       # 12 seed, entrambi i task, 3 modelli
python experiments/e1_summary.py                        # -> experiments/logs/v2/e1_summary.md
```
- `e1_summary.md` = una tabella per task: metriche in riga, modelli in colonna, media ± sd,
  migliore in grassetto, p-value di Welch di ogni GNN contro DistMult, miglior run, MRR "warm"
  (senza triple cold-start).
- Se lo script avvisa `config ... was not tuned under v2`, l'HPO v2 non è finito: DistMult viene
  saltato apposta (con il learning rate delle GNN sarebbe una baseline ingiusta).
- Per accorciare: `PROTOCOL=v2 RUNS=5 bash experiments/e1_main_training.sh`.

**Questo è il materiale principale della sezione.** Portami `e1_summary.md` (e `.csv`).

---

## 4. E3 — ablation v2 (dopo E1)

Modello di default R-GCN; se E1 dice che CompGCN è migliore usa `ABL_MODEL=compgcn`.
```bash
PROTOCOL=v2 bash experiments/e3_ablation.sh                         # Task A: componenti + contesto
PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh component    # Task B: componenti
python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_DTI    --out experiments/logs/v2/e3_DTI
python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_TREATS --out experiments/logs/v2/e3_TREATS
```
Componenti: focal off · adversarial off · 1 negativo · archi target di nuovo nel grafo · 50%
undersampling. Contesto (solo DTI): core_ppi / no_ppi / no_go / no_pathway / no_drugctx. 5 seed,
test appaiati con correzione di Holm. Portami i due `ablation_summary.md`.

---

## 5. E4 — repurposing e revisione esperta (con il modello migliore di E1)

```bash
ls -td models/dti_pkt_taskA_dti*  | head      # cartelle di E1, Task A (una per modello)
ls -td models/treats_pkt_taskB_treats* | head
ls models/<cartella>                           # il modello si riconosce dai file: rgcn_run*.pt / compgcn_run*.pt / distmult_run*.pt
bash experiments/e4_repurposing.sh A models/<cartella_task_A>
bash experiments/e4_repurposing.sh B models/<cartella_task_B>
```
`drug_eval.py` ora ricostruisce da solo split, grafo e config del run scelto (prima usava sempre
la config batterica e la relazione `TARGET`, e sui modelli PKT falliva). Per il foglio per
l'esperto: `experiments/README.md`, sezione "E4 in practice".

**Controllo obbligatorio sul pool dei candidati.** Di default E4 classifica *tutte* le proteine
(o malattie) del grafo, ma in training i negativi e la valutazione usano solo i nodi che compaiono
nella relazione target: gli altri non vengono mai "abbassati" e possono salire in cima al ranking
senza motivo biologico. Nell'output aggregato guarda `top20_outside_relation_pool`:
- se è basso → va bene il pool completo;
- se è alto (gran parte dei top-20 sono nodi mai visti nella relazione) → rilancia con
  `CANDIDATE_POOL=relation bash experiments/e4_repurposing.sh A models/<cartella>` e usa quel ranking
  (o riportali entrambi) per la revisione esperta.

---

## Cosa riportare indietro

1. `experiments/protocol_compare_summary.md` (E0)
2. `experiments/logs/v2/e1_summary.md` + `.csv` (E1) ← priorità
3. `experiments/logs/v2/e3_*/ablation_summary.md` (E3)
4. le cartelle `models/.../drug_eval_results/` dei modelli usati in E4
