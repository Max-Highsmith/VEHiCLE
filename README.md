# VEHiCLE Variationally Encoded Hi-C Loss Enhancement

This directory contains code used in running experiments for the paper:

**Highsmith, M. & Cheng, J. VEHiCLE: a Variationally Encoded Hi-C Loss Enhancement algorithm. doi:10.1101/2020.12.07.413559.**

## 1. Contact:
	Max Highsmith
	Department of Computer Science
	Email mrh8x5@mail.missouri.edu

## 2. Content of Folders:
	Data:    The Raw Data, Data Loaders, and preprcoessing scripts
	Models:  Pytorch implementation of models used in experiments
	Weights: Trained weights of experiments
	Experiments: Scripts used to run experiments
	Fig_Scripts: Scripts used To generate Figures
	other_tools: tools built by other labs need to run experiments

## 3.   Hi-C Data used in this study:
	In our study we used Hi-C data from GSE63525.  Datasets are programatically downloaded and formatted via the dataloader objects (Data/<cell_line>_DataModule.py) but can be found in their raw format at
	*https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE63525*.


## 3. Usage
	
To view the interactive tunable Hi-C Contact matrix generating GUI run
> python Generative_GUI.py
![gui]("Utils/gui.png)
To use VEHiCLE to enhance your own Hi-C data format your data into a pytorch dataloader with dataset of shape (sample size, 1, 269, 269) and run
> python enhance_HiC.py 
after replacing the currently loaded dataset object with your revised dataset.




