# Task A drug--target relation from UniProt cross-references

Mode: all drugs with a cross-reference. Relation name: `DTI`.


## Pipeline coverage

| step | value |
|---|---:|
| Human reviewed proteins with a DrugBank cross-reference (UniProt) | 3,393 |
| Distinct DrugBank accessions cross-referenced | 8,514 |
| Of these, named in the DrugBank CSV | 7,865 |
| Drugs skipped as not approved | 0 |
| Drugs matched to a ChEBI compound of the KG | 2,910 |
| Proteins dropped (accession absent from the KG) | 4 |
| **Edges emitted** | **19,329** |
| — of which `DTI` (pharmacodynamic targets) | **10,305** (53.3%) |
| — of which `DRUG_ADME` (enzymes, transporters, carriers) | 9,024 (46.7%) |
| Distinct drugs / proteins in `DTI` | 2,487 / 2,188 |
| rule — GO: xenobiotic/transport/carrier | 7,495 |
| rule — GO: no ADME annotation | 7,493 |
| rule — drugbank: no ADME protein declared | 2,812 |
| rule — drugbank: no target declared | 1,529 |
| Edges already asserted as `CPI_BIOCHEM` by PheKnowLator | n/d (task graph not built yet) |

## Most connected drugs

| targets | drug |
|---:|---|
| 161 | copper(2+) |
| 115 | zinc acetate |
| 90 | enflurane |
| 88 | promethazine |
| 76 | artenimol |
| 64 | miconazole |
| 60 | ethanol |
| 57 | calcium phosphate |
| 46 | phenethyl isothiocyanate |
| 37 | (S)-nicardipine |
| 35 | butabarbital |
| 35 | lamotrigine |
| 35 | cannabidiol |
| 34 | bioallethrin |
| 34 | valproic acid |
| 34 | aripiprazole |
| 33 | amoxapine |
| 33 | zonisamide |
| 33 | topiramate |
| 32 | loxapine |

## Most targeted proteins (`DTI`)

| drugs | protein |
|---:|---|
| 115 | solute carrier family 22 member 8 (human) |
| 103 | histamine H1 receptor (human) |
| 99 | prostaglandin G/H synthase 1 (human) |
| 93 | muscarinic acetylcholine receptor M1 (human) |
| 89 | gamma-aminobutyric acid receptor subunit alpha-1 (human) |
| 84 | alpha-2A adrenergic receptor (human) |
| 84 | alpha-1B adrenergic receptor (human) |
| 82 | muscarinic acetylcholine receptor M2 (human) |
| 80 | 5-hydroxytryptamine receptor 1A (human) |
| 79 | muscarinic acetylcholine receptor M3 (human) |
| 78 | gamma-aminobutyric acid receptor subunit alpha-3 (human) |
| 78 | gamma-aminobutyric acid receptor subunit alpha-2 (human) |
| 77 | beta-2 adrenergic receptor (human) |
| 77 | gamma-aminobutyric acid receptor subunit alpha-5 (human) |
| 74 | 5-hydroxytryptamine receptor 2C (human) |
| 72 | beta-1 adrenergic receptor (human) |
| 72 | alpha-1D adrenergic receptor (human) |
| 70 | estrogen receptor (human) |
| 70 | gamma-aminobutyric acid receptor subunit alpha-6 (human) |
| 70 | gamma-aminobutyric acid receptor subunit alpha-4 (human) |

## Proteins moved to `DRUG_ADME`

| drugs | protein |
|---:|---|
| 773 | cytochrome P450 3A4 (human) |
| 431 | ATP-dependent translocase ABCB1 (human) |
| 398 | albumin (human) |
| 336 | cytochrome P450 2D6 (human) |
| 331 | cytochrome P450 2C9 (human) |
| 268 | cytochrome P450 1A2 (human) |
| 263 | cytochrome P450 2C19 (human) |
| 251 | cytochrome P450 3A5 (human) |
| 230 | cytochrome P450 2C8 (human) |
| 170 | cytochrome P450 2B6 (human) |
| 159 | broad substrate specificity ATP-binding cassette transporter ABCG2 (human) |
| 131 | solute carrier organic anion transporter family member 1B1 (human) |
| 131 | cytochrome P450 3A7 (human) |
| 130 | alpha-1-acid glycoprotein 1 (human) |
| 122 | cytochrome P450 2E1 (human) |
| 119 | solute carrier family 22 member 6 (human) |
| 104 | UDP-glucuronosyltransferase 1A1 (human) |
| 94 | bile salt export pump (human) |
| 92 | D(2) dopamine receptor (human) |
| 92 | alpha-1A adrenergic receptor (human) |

## What is lost, and where

- 5,575 cross-referenced drugs could not be matched to a ChEBI compound (name matching only: no synonyms, CAS or UNII). Examples: not in the DrugBank CSV; not in the DrugBank CSV; no ChEBI match for '4-{[(Z)-(5-oxo-2-phenyl-1,3-oxazol-4(5H)-ylidene)methyl]amino}butanoic acid'; no ChEBI match for '2-(4-HYDROXY-3-NITROPHENYL)ACETIC ACID'; no ChEBI match for '4-HYDROXY-3-NITROPHENYLACETYL-EPSILON-AMINOCAPROIC ACID ANION'.
- 4 UniProt accessions are not present as Protein Ontology nodes.

Edge list: `dataset/PKT_subgraphs/dti_drugbank_edges.tsv`.
