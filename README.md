# LM-compositionality

This repo contains the code for the paper "Are representations built from the ground up? An empirical examination of local composition in language models". 

## How to access datasets

### Penn Treebank

Please download a version of the Penn Treebank files on your own machine. Afterwards, you can run `src/generate_data_treebank.py` to generate the data files from the Penn Treebank.
The path to treebank files should look something like this: `[...]/treebank_3/parsed/mrg/`. The dataset is automatically sharded into 10 parts, so to build the full dataset, you need to run with -i=[0...9].

`python3 -m src.generate_data_treebank --model=[bert,roberta,deberta,gpt2] -i=[0...9] --layer=12 --emb_type=[CLS,avg] [--cuda]`

Afterwards, you can generate the embeddings with this script: 

`python3 -m src.calc_compositionality_scores -i=[0...9] --emb_type=[CLS,avg] --layer=12 --model=[bert, roberta,deberta,gpt2] --full`

The embeddings will be saved to `data/binary_child_embs_[model]_[i]_[emb_type]_full_True_layer_12.npz". They can be combined again through running `python3 src/utils/combine_data.py`.

### CHIP (Compositionality of Human-annotated Idiomatic Phrases)

To access this dataset, you can download the csv file at `data/qualtrics_results/chip_dataset.csv`. Running `python3 -m src.data.process_qualtrics_data` will show details such as human annotator agreements and Spearman correlations between human results
and model compositionality scores. There are 1001 phrases in total.

Each phrase is scored from 1 (not compositional) to 3 (fully compositional) by three annotators. Each row represents the judgments of a different annotator.
An empty value for an annotator means that the annotator thought that the phrase didn't make sense (these were ignored for the analysis in the paper).

## Training probes

To train approximative probes, you can run the following script:

`python3 -m src.models.fit_composition_functions [add,mult,w1,w2,linear,mlp,affine] --model=[bert,roberta,deberta,gpt2] --emb_type=[cls,avg] --use_binary --full [--use_control_task]`

`--use_control_task` is for the anisotropy control setting (predicting randomly selected vectors rather than the parent vector). For the normal task, you don't need to run with this flag.


## Producing figures

These scripts can be found in `src/visualizations`.
