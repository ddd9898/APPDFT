# APPDFT: Reproduction Guide

All training and testing data adopted by APPDTF can be found at [APPDFT：An antigen prediction method based on data fusion and transformer - Mendeley Data](https://data.mendeley.com/datasets/fwxg5mgntn/). To reproduce these datasets, original data should be obtained from the following website:

a. NetMHCpan_train.tar.gz and 36 MS ligands files from https://services.healthtech.dtu.dk/suppl/immunology/NAR_NetMHCpan_NetMHCIIpan/

b. Table S4 and Table S8 from [Frontiers | DeepHLApan: A Deep Learning Approach for Neoantigen Prediction Considering Both HLA-Peptide Binding and Immunogenicity (frontiersin.org)](https://www.frontiersin.org/articles/10.3389/fimmu.2019.02559/full)

c.  ori_test_cells.csv and sars_cov_2_result.csv from [DeepImmuno/reproduce/data at main · frankligy/DeepImmuno (github.com)](https://github.com/frankligy/DeepImmuno/tree/main/reproduce/data)

Please follow the guide below to replicate the experiments in the full text of APPDFT.

# 1. Generate training data

```console
python ./utils/generateBAdataset.py
```

> Data size of binding affinity(BA) data NetMHCpan4.1 is 208093
> 170 MHCs in BA data

```console
python ./utils/generateELIMdataset.py
```

> Data size of DeepHLApan IM training data is 32787
> 118 MHCs in IM data
> After filtering samples, Data size of DeepHLApan IM training data is 32114
> Data size of eluted ligand(EL) data NetMHCpan4.1 is 12868293
> 143 MHCs in ELSA data
> pos/neg=223828/3877324
> After filtering samples, Data size of eluted ligand(EL) data NetMHCpan4.1 is 4101152
> 
> pure_IM=31537,pure_EL=4100575,mix=577
> conflict_count=3,EL1IM0_count=220

```console
python ./utils/generateFiveFolds.py
python ./utils/generatePseudoSeq.py
```

# 2. Training

## 2.1 Train the APPDFT and baseline-ELIG model

a) Set the the LINK variable at line 239 of the main_train_Model.py file to True, run the following commonds to acquire the APPDFT model: 

```console
python  main_train_Model.py   --fold 0  --load True --index 0  &
python  main_train_Model.py   --fold 1  --load True --index 0  &
python  main_train_Model.py   --fold 2  --load True --index 0  &
python  main_train_Model.py   --fold 3  --load True --index 0  &
python  main_train_Model.py   --fold 4  --load True --index 0  &
```

b) Set the LINK variable at line 239 of the main_train_Model.py file to False, re-run the above commonds to acquire the baseline-ELIG model.

## 2.2 Train the baseline-EL  model

Run the following commonds to acquire the baseline-EL model:

```console
python  main_train_baseline.py   --fold 0  --load True --index 0  &
python  main_train_baseline.py   --fold 1  --load True --index 0  &
python  main_train_baseline.py   --fold 2  --load True --index 0  &
python  main_train_baseline.py   --fold 3  --load True --index 0  &
python  main_train_baseline.py   --fold 4  --load True --index 0  &
```

# 3. Generate testing data

```console
python ./utils/generateELtestingDataset.py
python ./utils/generateIMtestingdataset.py
```

# 4. Testing

Change the three variables in lines 15 to 17 of main_test.py to test the performance of our three models on all independent testing datasets.

```console
python main_test.py
```

Run the following command to analyze the test results:

```console
python reproduce/analyseELresults.py
python reproduce/analyseIMresults.py
```

# 5. Reproduce figures

a) Run the R script in the reproduce/R/ folder and the following Python script in the reproduce/ folder to visualize all comparisons on the independent EL testing dataset.

```console
python reproduce/plot_ELresults_Change.py
```

<img title="" src="./R/AUC.png" alt="Alternative text" width="205"><img title="" src="./R/AUC0.1.png" alt="Alternative text" width="204"><img src="./R/PPV.png" title="" alt="Alternative text" width="216">

<img title="" src="../output/figures/AUC compare on EL testing data.png" alt="Alternative text" width="171"><img title="" src="../output/figures/AUC compare on EL testing data(self compare3).png" alt="Alternative text" width="471">

b) Run the following Python scripts in the reproduce/ folder to visualize all comparisons on independent IM/TESLA/Sars-Cov2 testing datasets.

```console
python reproduce/plot_IMresults_PRorROCcurve.py
python reproduce/plot_IMresults_Bar.py
```

<img src="../output/figures/PR curve benchmark on IM testing data.png" title="" alt="Alternative text" width="247"><img src="../output/figures/PR curve benchmark on IM testing data(9-10mer).png" title="" alt="Alternative text" width="250">

<img title="" src="../output/figures/Recall and Precision curve benchmark on IM testing data.png" alt="Alternative text" width="165"><img title="" src="../output/figures/Recall and Precision curve benchmark on TESLA testing data.png" alt="Alternative text" width="160"><img title="" src="../output/figures/Recall and Precision curve benchmark on Sars-Cov2-con testing data.png" alt="Alternative text" width="159"><img title="" src="../output/figures/Recall and Precision curve benchmark on Sars-Cov2-un testing data.png" alt="Alternative text" width="157">

<img title="" src="../output/figures/PR curve benchmark on IM testing data(Self).png" alt="Alternative text" width="201"><img title="" src="../output/figures/PR curve benchmark on TESLA testing data(Self).png" alt="Alternative text" width="204"><img title="" src="../output/figures/ROC curve benchmark on SarsCov2-conData(Self).png" alt="Alternative text" width="204">