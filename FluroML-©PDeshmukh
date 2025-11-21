import os
import streamlit as st
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc

# Try to import streamlit_ketcher for drawing
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("streamlit-ketcher is not installed. Please install it to draw molecules.")
        return ""

st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FluroML: Molecular Fluorescence Predictor")

# Load Models
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# Feature functions
def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return dc.feat.CircularFingerprint(radius=3, size=1024).featurize([mol])[0]

def smiles_to_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return pd.DataFrame(dc.feat.MACCSKeysFingerprint().featurize([mol]))

def predict(model, features):
    import numpy as np
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return model.predict(X)[0]

def read_molecule_file(uploaded_file):
    content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    if uploaded_file.name.endswith(".smi"):
        return content.split()[0]
    if uploaded_file.name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(content)
        return Chem.MolToSmiles(mol) if mol else None
    if uploaded_file.name.endswith(".sdf"):
        mol = Chem.MolFromMolBlock(content.split("$$$$")[0])
        return Chem.MolToSmiles(mol) if mol else None
    return None

@st.cache_data
def load_dataset():
    return pd.read_csv("All Properties with Finguprints_3.csv")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ===================== TAB 1 =====================
with tab1:
    st.header("🔎 Fluorescence Classification")

    method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"))
    smiles = ""

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES")
    elif method == "Upload File":
        file = st.file_uploader("Upload molecule", type=["smi","mol","sdf"])
        if file:
            smiles = read_molecule_file(file)
    else:
        smiles = st_ketcher("")

    if smiles and model_fluorescence:
        features = smiles_to_morgan(smiles)
        if features is not None:
            result = predict(model_fluorescence, features)
            st.success("Fluorescent" if int(result) == 1 else "Non-Fluorescent")

# ===================== TAB 2 =====================
with tab2:
    st.header("📈 Absorption Max Prediction")

    method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"))
    smiles = ""
    solvent = st.text_input("Solvent SMILES", value="O")

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES")
    elif method == "Upload File":
        file = st.file_uploader("Upload molecule", type=["smi","mol","sdf"])
        if file:
            smiles = read_molecule_file(file)
    else:
        smiles = st_ketcher("")

    if smiles and solvent and model_regression:
        f1 = smiles_to_descriptors(smiles)
        f2 = smiles_to_descriptors(solvent)
        features = pd.concat([f1,f2], axis=1)
        pred = predict(model_regression, features)
        st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# ===================== TAB 3 =====================
with tab3:
    st.header("📈 Emission Max Prediction")

    method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"))
    smiles = ""
    solvent = st.text_input("Solvent SMILES", value="O")

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES")
    elif method == "Upload File":
        file = st.file_uploader("Upload molecule", type=["smi","mol","sdf"])
        if file:
            smiles = read_molecule_file(file)
    else:
        smiles = st_ketcher("")

    if smiles and solvent and model_emission:
        f1 = smiles_to_descriptors(smiles)
        f2 = smiles_to_descriptors(solvent)
        features = pd.concat([f1,f2], axis=1)
        pred = predict(model_emission, features)
        st.success(f"Predicted Emission Max: {pred:.2f} nm")

# ===================== TAB 4 =====================
with tab4:
    st.header("🔗 FRET Pair Analysis")

    method = st.radio("Donor Input:", ("SMILES Input", "Draw Molecule", "Upload File"))
    donor_smiles = ""

    if method == "SMILES Input":
        donor_smiles = st.text_input("Donor SMILES")
    elif method == "Upload File":
        file = st.file_uploader("Upload donor", type=["smi","mol","sdf"])
        if file:
            donor_smiles = read_molecule_file(file)
    else:
        donor_smiles = st_ketcher("")

    if donor_smiles and model_emission:
        df = load_dataset()
        f1 = smiles_to_descriptors(donor_smiles)
        f2 = smiles_to_descriptors("O")
        features = pd.concat([f1,f2], axis=1)
        donor_em = predict(model_emission, features)

        df["Δ"] = abs(df["AbsorptioMax (nm)"] - donor_em)
        st.table(df.sort_values("Δ").head(5))

st.write("---")
st.caption("FluroML-©PDeshmukh")
