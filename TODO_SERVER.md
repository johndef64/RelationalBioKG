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

### Stato al 21 settembre 2026

| Passo | Stato |
|---|---|
| 0-2 · dataset ricostruiti, smoke test | ✅ fatto |
| 3 · HPO `-v2b` (30 trial per modello su A, 15 su B) | ✅ fatto; config `PKT-DTI-best-v2b` e `PKT-TREATS-best-v2b` in `src/models_params.json` (ora anche nel repo) |
| 3b · E0 ladder completo sul Task A | ✅ fatto: M 0,476 → 0,648, il 78% del guadagno viene dal checkpoint su M. È nel paper (tabella `tab:ladder`) |
| 4 · E1, 3 modelli × 2 task × 12 seed | ✅ fatto il 21/09. Report: [`docs/report_E1.md`](docs/report_E1.md). Tabelle nel paper e nel capitolo di tesi |
| 5 · E3 ablazione (componenti + contesto, A e B) | Task A ✅ completo su un'altra macchina, in [`experiments/ablation/DTI_v1/`](experiments/ablation/DTI_v1/MANIFEST.md). Task B ⏳ da rifare intero su una macchina sola con lo script nuovo (§5); la corsa parziale del server (3/13) non si unisce |
| 6 · E4 revisione esperta, Task A, R-GCN | ✅ chiusa il 22/09: 22,2% plausibili nei top-20 contro 1,1% dei decoy; catene meccanicistiche estratte per tutte e 40 le coppie. Report: [`docs/report_E4.md`](docs/report_E4.md). Nel paper (§5.6) e nel capitolo. Il revisore va citato solo come *domain expert*, senza nome |
| convergence check CompGCN sul TREATS | facoltativo, dopo E3 (§7) |
| HPO `-v2` DTI e TREATS | superati: restano su W&B come documentazione, non vanno mescolati |

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

Usa la **seconda versione del tandem**, `e2_hpo_tandem2.sh`: stessa logica di prima (Task A poi
Task B in sequenza, poi l'estrazione automatica delle config), ma i due task non condividono più un
unico budget di trial. Non costano uguale e non hanno alle spalle la stessa storia.

```bash
bash experiments/e2_hpo_tandem2.sh
```
Tutto il resto è già dentro: suffisso `-v2b`, 30 trial per modello su Task A, 15 su Task B, tetto
500 epoche, patience 10 valutazioni, estrazione finale di `PKT-DTI-best-v2b` e `PKT-TREATS-best-v2b`.

Prima di lanciare, controlla la testata: deve stampare
`Task A: 30 trials/model -> 90 runs` e `Task B: 15 trials/model -> 45 runs`.

Varianti utili:
```bash
bash experiments/e2_hpo_tandem2.sh A                          # solo Task A (+ la sua estrazione)
PKT_HPO_RUNS_A=40 PKT_HPO_RUNS_B=20 bash experiments/e2_hpo_tandem2.sh   # budget diversi
PKT_HPO_MODELS="rgcn" bash experiments/e2_hpo_tandem2.sh A    # un modello solo, per provare
```

- tre modelli per task (rgcn, compgcn, distmult); progetti W&B nuovi
  `RelationalPKT-DTI-v2b-*` e `RelationalPKT-TREATS-v2b-*`;
- **perché 30 e 15.** I 112 trial vecchi dicevano dove stava l'ottimo, ma la griglia del learning
  rate è stata allargata a `3e-2` e `1e-1` (vedi sotto): quei due valori non li ha mai provati
  nessuno, e E0 dice che l'ottimo sta proprio lì. Su Task A quindi si **esplora**, non si rifinisce,
  e costa poco (10.305 archi bersaglio su 1,16 M: un trial R-GCN sono un paio di minuti). Su Task B
  il budget resta basso perché costa una decina di volte tanto (168.157 archi bersaglio su 1,87 M,
  e CompGCN a grafo pieno è il caso che rischia l'OOM) — pur avendo meno storia alle spalle, visto
  che il vecchio sweep si era fermato a 22 trial e solo su R-GCN.
- il vecchio `e2_hpo_tandem.sh` resta com'era (un solo `PKT_HPO_RUNS` per entrambi i task): non
  usarlo qui, o Task B si prende lo stesso budget di Task A.

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
(per Task B: `HPO_SUFFIX=-v2b PKT_HPO_RUNS=15 bash experiments/resume_hpo.sh B`) completa fino al
budget del task e salta i modelli già finiti; poi ripeti l'estrazione. In alternativa si può
rilanciare `e2_hpo_tandem2.sh A` (o `B`): lo sweep W&B riprende dallo stesso sweep id.

Nota memoria: CompGCN su TREATS a grafo pieno è il caso più pesante; i trial in OOM vengono
registrati come saltati e lo sweep continua.

---

## 3b. E0 ladder completo — attribuzione gradino per gradino (~30 min) — ✅ FATTO

> Risultato: v1 M 0,476 → v2 M 0,648 (MRR 0,076 → 0,317). Split fisso −0,001 (p = 0,95), checkpoint su
> M +0,135 (78%), negativi espliciti −0,002, grafo pieno +0,014 (8%), supervisione disgiunta +0,026
> (15%). Il testo sotto resta come documentazione di come è stato eseguito.

Riempie l'unico buco rimasto nella sezione metodologica del paper. Oggi il paper dice che il
protocollo consolidato porta M da 0,542 a 0,741, ma non sa dire **quale** dei cinque cambiamenti ha
prodotto quel salto: la frase è `\tbd{per-step attribution}`. Il motivo è che quel confronto non è
mai stato eseguito per intero, nemmeno sul bersaglio vecchio: girava in modalità `pair`, che misura
solo tre gradini (`v1`, `+ split fisso`, `v2`). I quattro intermedi non esistono.

```bash
TASKS=DTI CMP_MODELS=rgcn CMP_RUNS=3 CMP_CONFIG=PKT-DTI-best-v2b \
  CMP_DIR=experiments/logs/e0_ladder_v2b bash experiments/e0_protocol_compare.sh ladder
python experiments/protocol_compare_summary.py --logdir experiments/logs/e0_ladder_v2b --out experiments
```

- **va lanciato dopo lo step 3**, perché usa `PKT-DTI-best-v2b`, la config appena estratta;
- sei varianti per tre run, più le baseline. Sul Task A nuovo la variante `v2` ha girato in 38 s per
  run (il bersaglio è passato da 25.713 a 10.305 archi), quindi si sta sotto la mezz'ora. Sul
  bersaglio vecchio la stessa variante costava 183 s per run: non usare quelle stime;
- `CMP_DIR` nuovo apposta: il crash-resume di E0 salterebbe le varianti trovando i log di 2b.

Cosa guadagna il paper: i gradini si leggono come attribuzione. Il primo (`+ split fisso`) non cambia
il training, quindi il suo Δ misura solo quanta della varianza di v1 era rumore di split; gli ultimi
(`+ negativi espliciti`, `+ grafo pieno`, `+ supervisione disgiunta`) cambiano cosa il modello vede e
misurano un miglioramento reale. È la differenza fra "misuravamo male" e "il modello è migliorato", e
senza la scala non si può affermare né l'una né l'altra.

In più lo misura **sulla relazione farmacologica**, non su quella biochimica ereditata: il paper può
togliere la premessa "questo confronto è stato fatto prima dell'iniezione, i valori assoluti non sono
confrontabili".

Se vuoi anche CompGCN aggiungi `CMP_MODELS="rgcn compgcn"` (circa un'ora in più). Sul Task B il
ladder costa molte ore: non è previsto, e non serve alla tesi che la scala sia misurata due volte.

---

## 4. E1 — la tabella per la tesi — ✅ FATTO (21/09)

> **Risultato.** Task A: R-GCN batte il modello senza encoder su AUROC/AUPRC/M (0,684 contro 0,657) ma
> pareggia sull'MRR (0,408 contro 0,407, p = 0,83): il contesto aiuta a riconoscere una coppia
> plausibile, non a ordinare il bersaglio giusto. Task B: DistMult vince su tutto (MRR 0,692 contro
> 0,348 e 0,229). Analisi completa, controlli (cold-start, quasi-duplicati ChEBI, convergenza) e punti
> aperti in [`docs/report_E1.md`](docs/report_E1.md).
>
> **Da sapere:** CompGCN sul TREATS non è arrivato a convergenza in 1500 epoche (limite inferiore). Il
> seed 6 di R-GCN TREATS è durato 40 ore per il blocco del terminale (§ "Regole per i job lunghi"):
> il risultato è valido, il suo `train_time_sec` no.
>
> Modelli: `models/dti_pkt_taskA_dti.tsv_20260918_{130755,135629,153101}` (R-GCN, CompGCN, DistMult)
> e `models/treats_pkt_taskB_treats.tsv_{20260918_151621,compgcn_20260920_123201,distmult_20260920_212827}`.
> Le sei cartelle `dti_…_20260918_09*` sono smoke test e si possono cancellare
> (`python experiments/models_index.py --stale`). Il testo sotto resta come documentazione.

Un comando solo, `e1_run_v2b.sh`, che fa entrambi i task, tutti i modelli e il summary finale.
**Va lanciato dentro tmux**: sono molte ore e una disconnessione ucciderebbe il processo.

```bash
git pull
tmux new -s e1
bash experiments/e1_run_v2b.sh
```

Per staccarti: `Ctrl+b` poi `d`. Per rientrare: `tmux attach -t e1`.

Dentro ci sono già suffisso `-v2b`, 12 semi, tre modelli, 1500 epoche e la generazione della tabella.
Varianti utili:

```bash
bash experiments/e1_run_v2b.sh A                      # solo Task A, il task economico (~3 h)
E1_MODELS="rgcn distmult" bash experiments/e1_run_v2b.sh B
E1_DRY=1 bash experiments/e1_run_v2b.sh               # stampa controlli e piano, non allena niente
E1_RESUME=0 bash experiments/e1_run_v2b.sh            # rifà tutto ignorando i log esistenti
```

Tre cose che lo script fa e che il comando grezzo non faceva:

- **preflight**: prima di allenare qualunque cosa verifica che `PKT-DTI-best-v2b` e
  `PKT-TREATS-best-v2b` esistano con tutti e tre i modelli, che i dataset ci siano, e che il codice
  sia aggiornato. Quest'ultimo controllo è il più utile: se dimentichi il `git pull`, il preflight
  se ne accorge e si ferma, invece di allenare per sei ore con 5 negativi invece dei 10 scelti
  dall'HPO. Se qualcosa manca stampa cosa e non allena niente;
- **crash-resume**: una coppia (task, modello) il cui log registra già il run finale viene saltata,
  quindi dopo un crash o un OOM basta rilanciare lo stesso comando;
- **tolleranza ai guasti**: se un modello fallisce (CompGCN sul TREATS è il caso che rischia l'OOM)
  gli altri proseguono, e alla fine ti dice quali sono caduti.

Costo: Task A circa 3 ore, Task B fino a 25 nel caso peggiore, meno con l'early stopping.

Da controllare nei log: che `best_epoch` non sia di nuovo contro il tetto. Nell'HPO quasi tutti i
trial ci finivano, ed è il motivo delle 1500 epoche; se succede ancora, rilancia con `EPOCHS=3000`.

**È il materiale principale della sezione.** Riportami `e1_summary.md` e `.csv`.

Nella tabella, oltre alle metriche solite, trovi due colonne nuove: `dedup_MRR` e
`dedup_stereo_MRR`, cioè l'MRR calcolato escludendo dal test le triple il cui fatto è già presente
in training attraverso un nodo ChEBI quasi-duplicato (varianti di carica, idratazione e sale la
prima; anche gli enantiomeri la seconda). Sul Task A escludono 12 triple su 2.380, quindi saranno
identiche alle primarie; sul Task B escludono il 7,6% e il 16% del test, e la differenza fra le due
colonne **è** la misura della fuga.

---

## 5. E3 — ablation — Task A ✅ (`DTI_v1`), Task B ⏳

Modello R-GCN, il migliore di E1 sul Task A. E3 **non riusa** i modelli di E1: riaddestra da zero con
la stessa config (`PKT-<TASK>-best-v2b`), gli stessi seed (`BASE_SEED + i`) e lo stesso split. 13
varianti per task × 5 seed, tetto 1500 epoche come E1.

**Stato al 24/09.** Il Task A è completo in [`experiments/ablation/DTI_v1/`](experiments/ablation/DTI_v1/MANIFEST.md),
eseguito su un'altra macchina (non il server): 13/13 varianti, audit superato, risultati e limiti nel
manifesto. Le corse parziali del server (Task A 10/13 del 21/09, Task B 3/13) **non vanno unite** a
nessuna versione: se le copi, tienile a parte (`archives/e3_server_partial/`).

### Cosa è cambiato il 24/09 (serve `git pull`, a job fermi)

- **Una cartella versionata per ogni run.** Tutto quello che produce un'ablazione (log, checkpoint,
  riepilogo, manifesto con macchina, GPU e commit) finisce in `experiments/ablation/<TASK>_v<N>/`.
  Niente più in `models/` (riservato a E1) né in `experiments/logs/`. Dettagli:
  [`experiments/ablation/README.md`](experiments/ablation/README.md).
- **Ripresa automatica della versione giusta.** Rilanciando lo stesso comando si riprende l'ultima
  versione incompleta; se è completa se ne apre una nuova. `ABL_VERSION=v<N>` forza una versione,
  `ABL_NEW=1` ne apre sempre una nuova. Una versione ripresa con impostazioni diverse viene rifiutata.
- **Modalità deterministica attiva di default** (`--deterministic`, `src/deterministic_ops.py`): due
  esecuzioni della stessa variante con lo stesso seme sono identiche bit per bit, verificato su 30
  epoche del Task A, costo circa +7% di tempo. In `DTI_v1` non c'era, e il seme 4 di `comp_full` e
  `ctx_full`, stessa ricetta, dava MRR 0,356 e 0,386. `ABL_DETERMINISTIC=0` torna al vecchio percorso.
- **Il GPU finisce nei log** (`[i] GPU: …` all'avvio) e nel manifesto.
- **Una variante che fallisce non ferma più le altre**: viene segnalata alla fine e la versione resta
  senza `COMPLETE`, così il rilancio la riprende.
- **Pre-registrazione.** Alla creazione di una versione lo script vi copia
  `experiments/prereg/PREREGISTRATION_<TASK>.json`: confronti primari, direzione dei test, soglia di
  equivalenza. Il riepilogo la segue. **Non modificare quei file dopo aver lanciato**: valgono perché
  sono scritti prima dei risultati, e il commit ne certifica la data.
- **`experiments/ablation/` non è in git** (`.gitignore`), quindi sul server `DTI_v1` non c'è: il Task A
  va lanciato con **`ABL_VERSION=v2`** esplicito, altrimenti lo script lo chiamerebbe `DTI_v1`.
- **Task A con 10 semi** (`ABL_RUNS=10`): `DTI_v1` è il pilota, `DTI_v2` la conferma.

**Sul cluster iknos non si lancia niente da terminale** (il nodo di gestione non ha GPU) **e non si usa
tmux**: ogni corsa è un job Slurm (`docs/guida_all_uso_del_nuovo_server.md`). Dalla root del repo:

```bash
git pull
# 1. verifica del sistema nuovo (circa 40 minuti, un job): alla fine del .out una riga PASS/FAIL per test
sbatch experiments/slurm/check_new_system.sbatch
squeue -u $USER
tail -n 12 experiments/slurm/check-<jobid>.out

# 2. solo se i 5 test sono PASS: le due ablazioni, ognuna su un tipo di GPU fisso
sbatch --job-name=e3A --gres=gpu:rtx3090:1    --export=ALL,ABL_TASK=A,ABL_VERSION=v2,ABL_RUNS=10 experiments/slurm/e3_ablation.sbatch
sbatch --job-name=e3B --gres=gpu:rtx5000ada:1 --export=ALL,ABL_TASK=B,ABL_VERSION=v1             experiments/slurm/e3_ablation.sbatch
```

- **Il tipo di GPU è fissato apposta**: nella coda `low` un job può essere interrotto e rimesso in coda,
  e senza `--gres` esplicito potrebbe ripartire sull'altro nodo, mescolando hardware nella stessa
  versione.
- **Se un job viene interrotto o supera il tempo**, risottomettilo con **lo stesso identico comando**:
  riprende la stessa versione e salta le varianti già complete. `ABL_VERSION` è sempre esplicito per
  questo.
- Tutto l'output sta in `experiments/ablation/<TASK>_<versione>/`, compreso `driver.log` con i messaggi
  dello script; il `.out` di Slurm contiene solo le prime righe.

Il Task B va fatto **tutto su una macchina sola**, riferimenti compresi. Durata circa 30 ore.

```bash
V=experiments/ablation/TREATS_v1
cat $V/MANIFEST.md                                  # quando, dove, che GPU, che commit
for f in $V/logs/*.log; do printf "  %-45s %s/5\n" "$(basename $f)" "$(grep -c 'Completed run' $f)"; done
grep -l "CUDA out of memory\|Traceback" $V/logs/*.log
cat $V/summary/ablation_summary.md                  # rigenerato a fine di ogni lancio
```

**Controllo di coerenza:** con `--deterministic`, `comp_full` e `ctx_full` devono risultare
**identici** seme per seme: stesso grafo, stesso ordine delle righe, stessa ricetta. Se differiscono,
qualcosa non è deterministico e va segnalato prima di leggere i Δ.

Come leggere: ogni variante si confronta col riferimento della sua famiglia (`comp_full` o
`ctx_full`) dentro la stessa versione. Guarda ΔAUROC oltre a ΔMRR: E1 dice che sul Task A il
contesto agisce sulla discriminazione più che sul ranking.

Le due varianti di contesto che rispondono a una domanda biologica:
- Task A `no_biochem`: la biochimica di PheKnowLator aiuta a predire i bersagli farmacologici?
  Risposta di `DTI_v1`: no (+0,002 di MRR).
- Task B `no_pharma`: quanto aggiunge il livello farmacologico alla predizione dell'indicazione?
  (prima dell'iniezione solo il 4,9% dei farmaci con un `TREATS` aveva un bersaglio molecolare, ora
  il 34,3%).

---

## 6. E4 — repurposing e revisione esperta — ✅ Task A fatto (22/09)

> Dal 21/09 E4 del Task A si fa **sul PC locale**, con il checkpoint R-GCN migliore di E1
> (`models/dti_pkt_taskA_dti.tsv_20260918_130755`, run 3), la coorte 1 di
> [`docs/coorte_validazione_taskA.md`](docs/coorte_validazione_taskA.md) e il foglio cieco con decoy di
> `expert_review_script.py`. La GPU del server resta libera per E3. I comandi sotto restano validi per
> chi volesse rifarlo sul server.
>
> **21/09: foglio generato.** 270 voci (9 composti × 20 predizioni, più 45 decoy casuali e 45 di centro
> classifica), pool `relation` (2.188 proteine). File in
> `models/dti_pkt_taskA_dti.tsv_20260918_130755/drug_eval_results/`:
> `expert_review_taskA_20260921_174910_BLIND.csv` (da compilare) e `…_KEY.csv` (**da non aprire**
> prima di aver finito). A revisione conclusa:
> `python expert_review_script.py aggregate <BLIND compilato> <KEY>`.
> Prima della generazione sono stati corretti due difetti dei decoy: potevano includere bersagli veri
> nascosti nel test, e quelli casuali venivano pescati fuori dal pool del modello.

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

## 7. Facoltativo, dopo E3 — convergenza di CompGCN sul TREATS

In E1 CompGCN sul Task B aveva ancora la validazione in salita a 1500 epoche (+0,025 di MRR nelle
ultime 250). Non cambia il confronto con DistMult, ma lascia aperto l'ordine fra le due GNN sul Task B.
Circa 6 ore per 3 seed:

```bash
# i log di convergenza vecchi (sweep -v2) verrebbero presi per lavoro già fatto: spostali
[ -d experiments/logs/v2/conv ] && mv experiments/logs/v2/conv experiments/logs/v2/conv_pre_v2b

tmux new -s conv   # dentro:
CONV_TASKS=TREATS CONV_MODELS=compgcn CONV_RUNS=3 CONV_EPOCHS=4000 CONV_GNN_LRS="" \
  bash experiments/convergence_check.sh
```
`CONV_GNN_LRS=""` fa girare solo il learning rate della config tunata: di default lo script prova
anche 0,03 e 0,1, che qui non servono. La config è `PKT-TREATS-best-v2b` grazie alla correzione di
`resolve_config`. Tabella in `experiments/logs/v2/conv/convergence_summary.md`.

---

## Regole per i job lunghi (dal blocco del 18-20/09)

Il seed 6 di R-GCN TREATS è rimasto fermo 40 ore. Gli script scrivevano con `tee` anche sul terminale
VS Code; a PC spento il terminale non veniva più svuotato, il buffer si riempiva e il training si
bloccava dentro una `write()`. È ripartito da solo alla riaccensione.

- **Corretto nel codice:** `run_logged` in `experiments/config.sh` scrive solo su file, e un `tail`
  separato mostra l'output. Se il terminale si blocca, si blocca solo il `tail`. Ogni riga `[val]`
  ha ora l'orario, e `e1_progress.sh` mostra la colonna `MIN/MAX m` con `!!` se un seed dura più di 4
  volte il più veloce.
- **Mai tmux.** Sul cluster iknos i job lunghi si sottomettono con `sbatch` (vedi §5 ed
  `experiments/slurm/`): girano da soli anche a browser e PC spenti. Le istruzioni con tmux più sopra
  (§4 E1, §7) sono della vecchia macchina e restano solo come storia.
- **Mai `git pull` mentre uno script `.sh` è in esecuzione**: bash legge lo script a pezzi e, se il
  file cambia sotto di lui, riprende dal punto sbagliato. Il pull si fa a job finito.

---

## Cosa riportare indietro

1. ~~`experiments/logs/v2/e1_summary.md` + `.csv`~~ ✅ ricevuti
2. ~~Task A~~ ✅ `experiments/ablation/DTI_v1/`. Per il Task B: l'intera cartella
   `experiments/ablation/TREATS_v<N>/`, senza `models/` (basta manifesto, log e `summary/`) ← **prossima priorità**
3. `python experiments/models_index.py --current-only` sul server, per sapere cosa tenere in `models/`

## Se qualcosa non torna

| Sintomo | Cosa fare |
|---|---|
| lo smoke test fallisce | guarda il log del passo in `experiments/logs/smoke/`, non proseguire |
| `missing dti_drugbank_edges.tsv` | hai saltato il passo 1: `python analysis/10_build_dti_drugbank.py` |
| UniProt non risponde | il file `dataset/DRUGBANK/uniprot_human_drugbank.tsv` è la cache: se c'è, lo script non scarica nulla |
| `config ... was not tuned under v2` in E1 | manca la config `-v2b`: rifai l'estrazione del passo 3, o controlla di aver passato `CFG_A`/`CFG_B` |
| OOM su CompGCN/TREATS | è il caso più pesante: `RUNS=5`, oppure escludi compgcn con `MODELS="rgcn distmult"` |
