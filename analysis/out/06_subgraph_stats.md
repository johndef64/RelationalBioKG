# PKT subgraphs — build statistics

INCLUDE_VARIANT = False · pharmacological layer = YES

Task A target = `DTI`, drug--target edges injected from DrugBank through the cross-references published by UniProt (`analysis/10_build_dti_drugbank.py`). PheKnowLator's own chemical--protein edges are biochemical (Reactome, UniProt catalysts, CTD) and are kept as context under the name `CPI_BIOCHEM`; see `TICKET_01_DTI_drug_scope.md`.

## De-duplicated edges per relation (one direction kept)

| relation | edges | origin | in A | in B | in unified |
|---|---:|---|:--:|:--:|:--:|
| COMPOUND_GO | 354,353 | PKT | ✓ | ✓ | ✓ |
| COMPOUND_PATHWAY | 29,860 | PKT | ✓ | ✓ | ✓ |
| CPI_BIOCHEM | 25,713 | PKT | ✓ | ✓ | ✓ |
| DISEASE_PHENOTYPE | 427,157 | PKT |  | ✓ | ✓ |
| DRUG_ADME | 9,024 | DrugBank/UniProt | ✓ | ✓ | ✓ |
| DTI | 10,305 | DrugBank/UniProt | ✓ | ✓ | ✓ |
| GDA | 12,757 | PKT |  | ✓ | ✓ |
| GDA_DYSFUNCTION | 4,494 | PKT |  | ✓ | ✓ |
| GENE_PATHWAY | 104,678 | PKT |  | ✓ | ✓ |
| GENE_PRODUCT | 19,478 | PKT | ✓ | ✓ | ✓ |
| PPI | 308,704 | PKT | ✓ | ✓ | ✓ |
| PROTEIN_GO_COMPONENT | 82,463 | PKT | ✓ | ✓ | ✓ |
| PROTEIN_GO_FUNCTION | 69,726 | PKT | ✓ | ✓ | ✓ |
| PROTEIN_GO_PROCESS | 129,189 | PKT | ✓ | ✓ | ✓ |
| PROTEIN_PATHWAY | 117,179 | PKT | ✓ | ✓ | ✓ |
| TREATS | 168,157 | PKT |  | ✓ | ✓ |

## Per-subgraph totals

### pkt_taskA_dti  — TASK A — predict DTI (drug->protein)

- file: `dataset/PKT_subgraphs/pkt_taskA_dti.tsv.zip`
- relations: ['COMPOUND_GO', 'COMPOUND_PATHWAY', 'CPI_BIOCHEM', 'DRUG_ADME', 'DTI', 'GENE_PRODUCT', 'PPI', 'PROTEIN_GO_COMPONENT', 'PROTEIN_GO_FUNCTION', 'PROTEIN_GO_PROCESS', 'PROTEIN_PATHWAY']
- **edges: 1,155,994**
- nodes per type: {'Compound': 8918, 'GO': 18881, 'Pathway': 2537, 'Protein': 19624, 'Gene': 19273}
- total nodes: 69,233
- edge origin: PKT: 1,136,665 (98.33%) · DrugBank/UniProt: 19,329 (1.67%)

### pkt_taskB_treats  — TASK B — predict TREATS (chemical->disease)

- file: `dataset/PKT_subgraphs/pkt_taskB_treats.tsv.zip`
- relations: ['COMPOUND_GO', 'COMPOUND_PATHWAY', 'CPI_BIOCHEM', 'DISEASE_PHENOTYPE', 'DRUG_ADME', 'DTI', 'GDA', 'GDA_DYSFUNCTION', 'GENE_PATHWAY', 'GENE_PRODUCT', 'PPI', 'PROTEIN_GO_COMPONENT', 'PROTEIN_GO_FUNCTION', 'PROTEIN_GO_PROCESS', 'PROTEIN_PATHWAY', 'TREATS']
- **edges: 1,873,237**
- nodes per type: {'Compound': 10733, 'GO': 18881, 'Pathway': 2537, 'Protein': 19624, 'Disease': 13673, 'Phenotype': 10051, 'Gene': 19612}
- total nodes: 95,111
- edge origin: PKT: 1,853,908 (98.97%) · DrugBank/UniProt: 19,329 (1.03%)

### pkt_unified  — UNIFIED — both targets (multi-task)

- file: `dataset/PKT_subgraphs/pkt_unified.tsv.zip`
- relations: ['COMPOUND_GO', 'COMPOUND_PATHWAY', 'CPI_BIOCHEM', 'DISEASE_PHENOTYPE', 'DRUG_ADME', 'DTI', 'GDA', 'GDA_DYSFUNCTION', 'GENE_PATHWAY', 'GENE_PRODUCT', 'PPI', 'PROTEIN_GO_COMPONENT', 'PROTEIN_GO_FUNCTION', 'PROTEIN_GO_PROCESS', 'PROTEIN_PATHWAY', 'TREATS']
- **edges: 1,873,237**
- nodes per type: {'Compound': 10733, 'GO': 18881, 'Pathway': 2537, 'Protein': 19624, 'Disease': 13673, 'Phenotype': 10051, 'Gene': 19612}
- total nodes: 95,111
- edge origin: PKT: 1,853,908 (98.97%) · DrugBank/UniProt: 19,329 (1.03%)

