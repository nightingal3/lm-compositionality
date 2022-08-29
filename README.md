# LM-compositionality

This repo contains the code for the paper "Are representations built from the ground up? An empirical examination of local composition in language models". 

## How to access datasets

### Penn Treebank

Please download a version of the Penn Treebank files on your own machine. Afterwards, you can run `src/generate_data_treebank.py` to generate the data files from the Penn Treebank.
The path to treebank files should look something like this: `.../treebank_3/parsed/mrg/`. The dataset is automatically sharded into 10 parts, so to build the full dataset,
you need to run with -i=[0...9].

`python3 src.generate_data_treebank --model=[bert,roberta,deberta,gpt2] -i=[0...9] --layer=12 --emb_type=[CLS,avg] [--cuda]`

Afterwards, you can generate the embeddings with this script: 

``

### CHIP (Compositionality of Human-annotated Idiomatic Phrases)

To access this dataset, you can download the csv file at `data/qualtrics_results/chip_dataset.csv`. Running `python3 -m src.data.process_qualtrics_data` will show details such as human annotator agreements and Spearman correlations between human results
and model compositionality scores. There are 1001 phrases in total.

Each phrase is scored from 1 (not compositional) to 3 (fully compositional) by three annotators. Each row represents the judgments of a different annotator.
An empty value for an annotator means that the annotator thought that the phrase didn't make sense (these were ignored for the analysis in the paper).

## Training probes


## Producing figures
