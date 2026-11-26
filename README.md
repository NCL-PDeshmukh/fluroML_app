
**FluroML**:**Robust Prediction of Fluorescence Properties using Machine Learning Approaches**

Machine-learning prediction of fluorescence properties: fluorescent/non-fluorescent classification, absorption and emission maxima, solvent effects, and FRET pair screening.

**Live app**: https://fluromlapp-eavu9hwqdwgwft2ruvquwf.streamlit.app/

Overview
This project leverages machine learning (ML) to predict fluorescence properties of molecules with high accuracy and robustness. Fluorescence plays a critical role in bioimaging, sensing, and material science. Traditional experimental methods are costly and time-consuming, making computational prediction a powerful alternative.
Key Features

**Reproducible ML Pipeline**: End-to-end workflow for data preprocessing, model training, and evaluation.

**Interactive Streamlit App**: Real-time predictions and visualization for user-friendly exploration.

**Advanced Functionalities:**
Functional Group Pair Classification: Identify and classify key functional group pairs influencing fluorescence.
Fluorescence Classification: Predict whether a molecule exhibits fluorescence behavior.
Spectral Properties Prediction:
λ_max (Absorption Maximum) prediction.
Emission Maximum prediction.
Solvent Effect Modeling: Incorporate solvent polarity and environment effects for accurate predictions.

<img width="3902" height="1600" alt="eToC_8-4" src="https://github.com/user-attachments/assets/222a709d-717b-43cf-a219-fc3378ec03b4" />


**Install**
bash
git clone https://github.com/NCL-PDeshmukh/fluroML_app.git
cd fluroML_app
conda env create -f environment.yml
conda activate fluroml
Or with pip on Python 3.11

**bash**
pip install -r requirements.txt
Run
bash
streamlit run "FluroML-©PDeshmukh.py"

**Software versions**
Python 3.11 · scikit-learn 1.1.2 · rdkit-pypi 2023.3.1b1 · deepchem 2.8.0 · numpy 1.23.5 · pandas 1.5.3 · joblib 1.2.0 · streamlit 1.27.0

**Models**
File	Predicts	Model
best_classifier_compatible.joblib	fluorescent / non-fluorescent	Morgan + RF
new_best_regressor_compatible.joblib	absorption maximum	MACCS + RF
best_regressor_emission_compatible.joblib	emission maximum	MACCS + RF

**Data**
FluroML.csv (curated dataset) · Classification_Dataset_11,986.csv · train_set.csv / test_set.csv (70/30 split) · TRAIN_Complete_All_Features.rar / TEST_Complete_All_Features.csv · Morgan,MACCs and RDkit feature.npy.rar · External_Validation_1500.csv
