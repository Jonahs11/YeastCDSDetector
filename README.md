# YeastCDSDetector

## Overview
This repository contains code and data-processing scripts for fitting Hidden Markov Models (HMMs) to genomic signal data from Saccharomyces cerevisiae chromosome XVI. The workflow includes preparing raw data, converting it into a usable format, and fitting HMMs with either 2 or 4 hidden states. Result files are stored in the results.

## Structure
- `create_useable_format.ipynb` — prepares raw data  
- `fit_hmm.py` — fits a 2-state HMM  
- `fit_hmm_4states.py` — fits a 4-state HMM  
- `fit_hmm.ntbk.ipynb` — notebook version of the HMM workflow  
- `results/` — output HMM results  
- `yeast_chr16/` — input data

## Usage
1. Run `create_useable_format.ipynb` to generate clean input data.  
2. Run `fit_hmm.py` or `fit_hmm_4states.py` to fit the models.  
3. Results appear in the `results/` folder.
