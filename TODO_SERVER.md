# TODO — sessione sul server (ripartenza dopo il TICKET 01)

Obiettivo della sessione: ricostruire i dataset con il livello farmacologico, verificare che tutto
giri, rifare l'HPO del **Task A** (il bersaglio è cambiato) e arrivare alla tabella di E1 per la tesi.

Tutti i comandi dalla root del repo, con l'env `gnn` attivo. Gli script `experiments/*.sh` fissano
da soli `PYTHONHASHSEED=0`. La versione precedente di questo file è in
[`TODO_SERVER_legacy.md`](TODO_SERVER_legacy.md).

---

## Cosa è cambiato e perché (in breve)

La relazione bersaglio del Task A ereditata da PheKnowLator era **biochimica, non farmacologica**: i
composti più connessi erano idrogenione, acqua, ATP e magnesio, e coprivano il 31,4% degli archi.
Predirla significa predire biochimica nota, non medicina. Dettagli e alternative scartate:
[`TICKET_01_DTI_drug_scope.md`](TICKET_01_DTI_drug_scope.md).

I grafi ora tengono separati tre livelli di evidenza:

| Relazione | Contenuto | Origine | Ruolo |
|---|---|---|---|
| `DTI` | bersaglio farmacodinamico, 10.305 archi | DrugBank via UniProt | **bersaglio Task A**, contesto Task B |
| `DRUG_ADME` | enzimi, trasportatori, proteine plasmatiche, 9.024 archi | DrugBank via UniProt | contesto |
| `CPI_BIOCHEM` | substrati, cofattori, catalisi, 25.713 archi | PheKnowLator | contesto |

| Cosa | Stato |
|---|---|
| HPO v2 **TREATS** | **valido**, il bersaglio non è cambiato — tenerlo |
| HPO v2 **DTI** | da rifare: era sul bersaglio biochimico |
| E0 (confronto protocolli) | valido come studio di protocollo, non va rifatto |
| E1, E3, E4 | da eseguire sui grafi nuovi |

---

## 0. Pulizia e aggiornamento del codice

**Non cancellare** `dataset/PKT/` (il KG grezzo, ~5 GB) né `dataset/DRUGBANK/`: servono per
ricostruire. Cancella solo i dataset **generati**.

```bash
# 0a. se l'HPO del TREATS sta ancora girando, aspetta che finisca
ps aux | grep -c "[t]uning_hyperparameter.py"      # deve stampare 0

# 0b. salva ciò che non è in git e non è rigenerabile
tar czf ~/pkt_backup_$(date +%F).tgz experiments/logs docs setup_wandb.sh src/models_params.json

# 0c. elimina i dataset generati
rm -rf dataset/PKT_subgraphs/ablation
rm -f  dataset/PKT_subgraphs/pkt_task*.tsv.zip dataset/PKT_subgraphs/pkt_unified.tsv.zip \
       dataset/PKT_subgraphs/dti_drugbank_edges.tsv

# 0d. aggiorna il codice
git pull            # se protesta per modifiche locali: git checkout -- <file> e ripeti
ls dataset/PKT dataset/DRUGBANK    # devono esserci ancora nodes.zip/edges.zip e drug-bank-5110.zip
```

`dataset/PKT_subgraphs/node_labels.tsv` puoi tenerlo: non cambia. Se lo cancelli, si rigenera con
`python analysis/08_build_node_labels.py`.

---

## 1. Ricostruzione dei dataset

```bash
python analysis/10_build_dti_drugbank.py      # livello farmacologico (scarica da UniProt, ~1 min)
python analysis/06_build_subgraphs.py         # grafi dei due task (~5 min, rilegge il KG grezzo)
python analysis/07_build_ablation_subgraphs.py --task A
python analysis/07_build_ablation_subgraphs.py --task B
```

**Numeri attesi** (se non tornano, fermati):

| | valore |
|---|---|
| archi iniettati | 19.329 → `DTI` 10.305 + `DRUG_ADME` 9.024 |
| Task A | 1.155.994 archi, 69.233 nodi, bersaglio 10.305 |
| Task B | 1.873.237 archi, 95.111 nodi, bersaglio 168.157 |
| origine Task A | PheKnowLator 98,33% · DrugBank/UniProt 1,67% |
| ablation | 7 varianti per task |

Report: `analysis/out/10_dti_drugbank_report.md` e `analysis/out/06_subgraph_stats.md`.

---

## 2. Verifica che tutto giri (obbligatoria, pochi minuti)

```bash
SMOKE_GPU=1 bash experiments/smoke_test.sh
```
Esegue un run minimo di ogni passo: caricamento dati, training dei tre modelli su entrambi i task,
grafi di ablation, baseline di popolarità, un modello salvato più `drug_eval`, script di riepilogo.
**Deve chiudere con `FAIL: 0`.** I log dei passi falliti stanno in `experiments/logs/smoke/`.

---

## 2b. Valutazione rapida del nuovo Task A (prima di investire nell'HPO)

Serve a sapere subito se il bersaglio nuovo è imparabile, **prima** di spendere ore di HPO. Usa la
config migliore disponibile (quella tunata sotto v2 sul vecchio bersaglio: imperfetta ma ragionevole)
e confronta R-GCN con il pavimento di popolarità e con DistMult su una piccola griglia di learning
rate. Circa un'ora.

```bash
TASKS=DTI CMP_MODELS=rgcn CMP_RUNS=3 CMP_CONFIG=PKT-DTI-best-v2 \
  bash experiments/e0_protocol_compare.sh quick
python experiments/protocol_compare_summary.py     # -> experiments/protocol_compare_summary.md
```

Come leggere la tabella:

| Esito | Significato | Cosa fare |
|---|---|---|
| R-GCN e DistMult **sopra** la popolarità, MRR nell'ordine di 0,2–0,5 | il bersaglio è imparabile | prosegui con l'HPO |
| entrambi **vicini** alla popolarità | il segnale è quasi solo grado dei nodi | fermati e portami la tabella: forse servono più archi bersaglio (mapping per sinonimi/CAS) |
| metriche quasi perfette (MRR > 0,9) | sospetta ridondanza fra contesto e bersaglio | fermati: `DRUG_ADME` e `DTI` vengono dalla stessa fonte, va verificato che non ci sia sovrapposizione |

Riferimenti dal vecchio bersaglio (biochimico), utili solo come ordine di grandezza: popolarità
M 0,478 · MRR 0,117 — R-GCN v2 M 0,741 · MRR 0,436 — DistMult M 0,795 · MRR 0,598.

---

## 3. HPO del Task A (il bersaglio è nuovo)

Gli sweep vecchi (`RelationalPKT-DTI-v2-*`) riguardano il bersaglio biochimico: **non vanno
mescolati**. Si usa quindi un suffisso nuovo, `-v2b`.

```bash
HPO_SUFFIX=-v2b PKT_HPO_RUNS=15 bash experiments/e2_hpo_sweep.sh A
python experiments/get_best_hpo_config.py --task DTI --suffix=-v2b --write
```
- 15 trial per modello invece di 30: i 90 trial precedenti dicono già dove sta l'ottimo (lr al bordo
  superiore, un solo layer di convoluzione, 10 negativi per positivo);
- tetto 500 epoche e patience 10 valutazioni sono i default di `e2_hpo_sweep.sh` sotto v2;
- l'estrazione scrive `PKT-DTI-best-v2b` in `src/models_params.json`.

**Check:** il comando deve stampare `[RelationalPKT-DTI-v2b-<modello>] ranking by
'best_val_mixed_metric'` per rgcn, compgcn e distmult. Poi:
```bash
python -c "import json;d=json.load(open('src/models_params.json'));print({k:list(v) for k,v in d.items() if 'best-v2' in k})"
```
deve mostrare sia `PKT-DTI-best-v2b` sia `PKT-TREATS-best-v2` con tre modelli ciascuno.

Se lo sweep si interrompe: `HPO_SUFFIX=-v2b PKT_HPO_RUNS=15 bash experiments/resume_hpo.sh A`
(completa fino a 15 trial per modello, salta i modelli già finiti), poi ripeti l'estrazione.

---

## 4. E1 — la tabella per la tesi

Il Task A usa la config nuova, il Task B quella già tunata.

```bash
# Task A (config nuova, suffisso -v2b)
CFG_A=PKT-DTI-best-v2b PROTOCOL=v2 EPOCHS=1500 TASKS=A bash experiments/e1_main_training.sh

# Task B (config -v2 già presente, risolta da sola)
PROTOCOL=v2 EPOCHS=1500 TASKS=B bash experiments/e1_main_training.sh

python experiments/e1_summary.py        # -> experiments/logs/v2/e1_summary.md
```
- 12 seed, tre modelli (R-GCN, CompGCN, DistMult), split fisso;
- tetto alto apposta: nell'HPO quasi tutti i trial finivano sul tetto delle epoche. Controlla nei log
  che `best_epoch` non sia di nuovo al tetto; se lo è, rilancia con `EPOCHS=3000`;
- per una passata rapida: aggiungi `RUNS=5`.

**È il materiale principale della sezione.** Riportami `e1_summary.md` e `.csv`.

---

## 5. E3 — ablation (dopo E1)

Modello di default R-GCN; se E1 dice che CompGCN è migliore, usa `ABL_MODEL=compgcn`.

```bash
PROTOCOL=v2 bash experiments/e3_ablation.sh                        # Task A: componenti + contesto
PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh component   # Task B: componenti
python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_DTI    --out experiments/logs/v2/e3_DTI
python experiments/ablation_summary.py --logdir experiments/logs/v2/e3_TREATS --out experiments/logs/v2/e3_TREATS
```

Le due varianti di contesto nuove sono quelle che rispondono a una domanda biologica:
- Task A `no_biochem`: la biochimica di PheKnowLator aiuta a predire i bersagli farmacologici?
- Task B `no_pharma`: quanto aggiunge il livello farmacologico alla predizione dell'indicazione?
  (prima dell'iniezione solo il 4,9% dei farmaci con un `TREATS` aveva un bersaglio molecolare, ora
  il 34,3%).

---

## 6. E4 — repurposing e revisione esperta

```bash
ls -td models/dti_pkt_taskA_dti* | head       # cartelle di E1
ls -td models/treats_pkt_taskB_treats* | head
CANDIDATE_POOL=relation bash experiments/e4_repurposing.sh A models/<cartella_task_A>
CANDIDATE_POOL=relation bash experiments/e4_repurposing.sh B models/<cartella_task_B>
```

- usa **`CANDIDATE_POOL=relation`**: il pool completo contiene tutte le proteine del grafo, comprese
  quelle che non compaiono mai in una relazione farmacologica, e finirebbero in cima senza motivo;
  nell'output aggregato controlla comunque `top20_outside_relation_pool`;
- i composti da mostrare all'esperto vanno scelti **prima** di vedere le predizioni, dalle coorti
  richieste ai clinici (`docs/richiesta_candidati_validazione_*.md`): serve conoscenza esterna al
  grafo, altrimenti la validazione è circolare.

---

## Cosa riportare indietro

1. `analysis/out/06_subgraph_stats.md` e `analysis/out/10_dti_drugbank_report.md` (dataset ricostruiti)
2. `experiments/logs/v2/e1_summary.md` + `.csv` ← **priorità**
3. `experiments/logs/v2/e3_*/ablation_summary.md`
4. le cartelle `models/.../drug_eval_results/` usate in E4

## Se qualcosa non torna

| Sintomo | Cosa fare |
|---|---|
| lo smoke test fallisce | guarda il log del passo in `experiments/logs/smoke/`, non proseguire |
| `missing dti_drugbank_edges.tsv` | hai saltato il passo 1: `python analysis/10_build_dti_drugbank.py` |
| UniProt non risponde | il file `dataset/DRUGBANK/uniprot_human_drugbank.tsv` è la cache: se c'è, lo script non scarica nulla |
| `config ... was not tuned under v2` in E1 | manca la config: rifai l'estrazione del passo 3 |
| OOM su CompGCN/TREATS | è il caso più pesante: `RUNS=5`, oppure escludi compgcn con `MODELS="rgcn distmult"` |
