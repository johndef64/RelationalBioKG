# TODO — sessione sul server (ripartenza dopo il TICKET 01)

Obiettivo della sessione: ricostruire i dataset con il livello farmacologico, verificare che tutto
giri, rifare l'HPO di **entrambi i task** sui dati definitivi e arrivare alla tabella di E1 per la tesi.

Tutto ciò che viene prodotto da qui in avanti porta il suffisso **`-v2b`**: progetti W&B
`RelationalPKT-<TASK>-v2b-<modello>`, config `PKT-<TASK>-best-v2b`. I progetti `-v2` restano
intatti ma riguardano dati superati (il DTI aveva il bersaglio biochimico, il TREATS il grafo senza
livello farmacologico) e **non vanno mescolati** con i nuovi.

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
| HPO `-v2` **DTI** (90 trial) | superato: era sul bersaglio biochimico. Resta su W&B come documentazione |
| HPO `-v2` **TREATS** (22 trial R-GCN, M 0,746) | interrotto a un quarto; resta su W&B come riferimento |
| E0 (confronto protocolli) | valido come studio di protocollo, non va rifatto |
| HPO `-v2b`, E1, E3, E4 | da eseguire sui grafi nuovi |

Perché si rifà anche il TREATS: lo sweep era arrivato a 22 trial su 90 (CompGCN e DistMult non erano
nemmeno partiti), quindi doveva comunque girare per il grosso del lavoro. Tanto vale che giri sul
grafo definitivo, così i due task sono tunati sugli stessi dati e non serve dichiarare in nota che il
Task B è stato ottimizzato su un grafo diverso da quello usato in E1.

---

## 0. Pulizia e aggiornamento del codice

**Non cancellare** `dataset/PKT/` (il KG grezzo, ~5 GB) né `dataset/DRUGBANK/`: servono per
ricostruire. Cancella solo i dataset **generati**.

```bash
# 0a. ferma l'HPO del TREATS se sta ancora girando (i trial fatti restano su W&B)
pkill -f tuning_hyperparameter.py
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
# archivia i log di E0 sul dataset vecchio: il resume li scambierebbe per lavoro già fatto
mkdir -p experiments/logs/archive_e0_biochem
mv experiments/logs/e0/* experiments/logs/archive_e0_biochem/ 2>/dev/null

TASKS=DTI CMP_MODELS=rgcn CMP_RUNS=3 CMP_CONFIG=PKT-DTI-best-v2 CMP_DIR=experiments/logs/e0_v2b \
  bash experiments/e0_protocol_compare.sh quick
python experiments/protocol_compare_summary.py --logdir experiments/logs/e0_v2b --out experiments
```

**Controlla le prime righe:** deve allenare davvero. Se stampa `skip ... (already complete)` per
tutte le varianti, sta leggendo log vecchi: `CMP_DIR` punta a una cartella già usata.
La tabella riporta l'ora di fine e il tempo per run, usali per accorgertene.

Come leggere la tabella:

| Esito | Significato | Cosa fare |
|---|---|---|
| R-GCN e DistMult **sopra** la popolarità, MRR nell'ordine di 0,2–0,5 | il bersaglio è imparabile | prosegui con l'HPO |
| entrambi **vicini** alla popolarità | il segnale è quasi solo grado dei nodi | fermati e portami la tabella: forse servono più archi bersaglio (mapping per sinonimi/CAS) |
| metriche quasi perfette (MRR > 0,9) | sospetta ridondanza fra contesto e bersaglio | fermati: `DRUG_ADME` e `DTI` vengono dalla stessa fonte, va verificato che non ci sia sovrapposizione |

Riferimenti dal vecchio bersaglio (biochimico), utili solo come ordine di grandezza: popolarità
M 0,478 · MRR 0,117 — R-GCN v2 M 0,741 · MRR 0,436 — DistMult M 0,795 · MRR 0,598.

---

## 3. HPO `-v2b` — entrambi i task sui dati definitivi (il passo più lungo)

I due task si lanciano **separati**, con budget diversi (`e2_hpo_tandem.sh` userebbe un solo
`PKT_HPO_RUNS` per entrambi). Il tempo sta quasi tutto in Task B, quindi si spende dove costa poco.

```bash
# Task A — economico (10.305 archi bersaglio su 1,16 M): un trial R-GCN sono un paio di minuti
HPO_SUFFIX=-v2b PKT_HPO_RUNS=30 bash experiments/e2_hpo_sweep.sh A

# Task B — il pesante (168.157 archi bersaglio su 1,87 M, CompGCN a grafo pieno rischia l'OOM)
HPO_SUFFIX=-v2b PKT_HPO_RUNS=15 bash experiments/e2_hpo_sweep.sh B

# estrazione delle config migliori (il tandem la faceva da solo, qui va lanciata a mano)
python experiments/get_best_hpo_config.py --task DTI    --suffix=-v2b --write
python experiments/get_best_hpo_config.py --task TREATS --suffix=-v2b --write
```
- tre modelli per task (rgcn, compgcn, distmult), tetto 500 epoche e patience 10 valutazioni
  (default v2 di `e2_hpo_sweep.sh`);
- `PKT_HPO_RUNS` va passato sempre: lanciato da solo, `e2_hpo_sweep.sh` ha come default **100**
  trial per modello (era il tandem a portarlo a 30);
- progetti W&B nuovi: `RelationalPKT-DTI-v2b-*` e `RelationalPKT-TREATS-v2b-*`;
- **30 trial su Task A, 15 su Task B.** I 112 trial vecchi dicevano dove stava l'ottimo, ma la
  griglia del learning rate è stata allargata a `3e-2` e `1e-1` (vedi sotto): quei due valori non
  sono mai stati provati, e E0 dice che l'ottimo sta proprio lì. Su Task A quindi si esplora, non si
  rifinisce, e costa poco. Su Task B il budget resta basso perché è dieci volte più caro — pur
  avendo meno storia alle spalle (il vecchio sweep si era fermato a 22 trial e solo su R-GCN).

**Griglia del learning rate allargata** (`tuning_hyperparameter.py`, blocco `_V2`):
`[1e-3, 3e-3, 1e-2, 3e-2, 1e-1]`, prima si fermava a `1e-2`. In full-batch si fa un solo passo di
ottimizzatore per epoca, quindi il learning rate decide tutto: in E0 DistMult ha fatto M 0,300 →
0,346 → 0,531 passando da 1e-2 a 3e-2 a 1e-1, e R-GCN a 1e-3 stava ancora migliorando contro il
tetto delle epoche (best epoch 297/300). Senza questa modifica la baseline avrebbe cercato fino a
1e-1 e le GNN solo fino a 1e-2: passi dieci volte più corti per il modello che deve batterla.
**Serve un `git pull` sul server prima di lanciare.**

**Check:** i comandi devono stampare `[RelationalPKT-<TASK>-v2b-<modello>] ranking by
'best_val_mixed_metric'` per rgcn, compgcn e distmult. Poi:
```bash
python -c "import json;d=json.load(open('src/models_params.json'));print({k:list(v) for k,v in d.items() if k.endswith('-v2b')})"
```
deve mostrare `PKT-DTI-best-v2b` e `PKT-TREATS-best-v2b`, con tre modelli ciascuno.

Se lo sweep si interrompe: `HPO_SUFFIX=-v2b PKT_HPO_RUNS=30 bash experiments/resume_hpo.sh A`
(per Task B: `PKT_HPO_RUNS=15 ... resume_hpo.sh B`) completa fino al budget del task e salta i
modelli già finiti; poi ripeti l'estrazione.

Nota memoria: CompGCN su TREATS a grafo pieno è il caso più pesante; i trial in OOM vengono
registrati come saltati e lo sweep continua.

---

## 4. E1 — la tabella per la tesi

Entrambi i task usano le config `-v2b` appena estratte (`CFG_A` / `CFG_B` le forzano: `resolve_config`
da solo cercherebbe il suffisso `-v2`). Per allenare un task alla volta: `TASKS=A` oppure `TASKS=B`.

```bash
CFG_A=PKT-DTI-best-v2b CFG_B=PKT-TREATS-best-v2b PROTOCOL=v2 EPOCHS=1500 bash experiments/e1_main_training.sh
python experiments/e1_summary.py        # -> experiments/logs/v2/e1_summary.md
```
- 12 seed, tre modelli (R-GCN, CompGCN, DistMult), split fisso;
- tetto alto apposta: nell'HPO quasi tutti i trial finivano sul tetto delle epoche. Controlla nei log
  che `best_epoch` non sia di nuovo al tetto; se lo è, rilancia con `EPOCHS=3000`;
- per una passata rapida: aggiungi `RUNS=5`.

**È il materiale principale della sezione.** Riportami `e1_summary.md` e `.csv`.

---

## 5. E3 — ablation (dopo E1)

Modello di default R-GCN; se E1 dice che CompGCN è migliore, usa `ABL_MODEL=compgcn`. Anche qui la
config va forzata con `ABL_CONFIG`, altrimenti lo script risolve il suffisso `-v2`, cioè i dati vecchi.

```bash
ABL_CONFIG=PKT-DTI-best-v2b PROTOCOL=v2 bash experiments/e3_ablation.sh                      # Task A
ABL_CONFIG=PKT-TREATS-best-v2b PROTOCOL=v2 ABL_TASK=B bash experiments/e3_ablation.sh component   # Task B
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
| `config ... was not tuned under v2` in E1 | manca la config `-v2b`: rifai l'estrazione del passo 3, o controlla di aver passato `CFG_A`/`CFG_B` |
| OOM su CompGCN/TREATS | è il caso più pesante: `RUNS=5`, oppure escludi compgcn con `MODELS="rgcn distmult"` |
