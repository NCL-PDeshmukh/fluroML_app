import streamlit as st
import joblib
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem

# ============================================================
# 🔇  REMOVE ALL WARNINGS (RDKit + DeepChem + TF + Torch + Jax)
# ============================================================
import warnings
warnings.filterwarnings("ignore")

from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
RDLogger.DisableLog('rdMolDraw*')

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPLOG_LEVEL"] = "0"
os.environ["DEEPLOG_DISABLE"] = "1"
os.environ["DC_SILENCE_LOGGING"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RDKIT_SILENCE_DEPRECATION_WARNINGS"] = "1"

# PIL SAFE RENDERER (NO X11)
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io

def safe_mol_image(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    d = rdMolDraw2D.MolDraw2DCairo(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    png = d.GetDrawingText()
    return Image.open(io.BytesIO(png))

# Try to import streamlit_ketcher
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("Ketcher not installed.")
        return ""

# ============================================================
# STREAMLIT CONFIG
# ============================================================
st.set_page_config(page_title="FluroML 2026", layout="wide")

# ============================================================
# LOAD MODELS
# ============================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model load error {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# ============================================================
# FEATURE GENERATORS
# ============================================================
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES")
        return None

    feat = dc.feat.CircularFingerprint(size=1024, radius=3)
    try:
        return feat.featurize([mol])[0]
    except:
        return None

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        st.error("Invalid SMILES")
        return None
    feat = dc.feat.MACCSKeysFingerprint()
    return pd.DataFrame(feat.featurize([mol]))

# ============================================================
# GENERIC PREDICT
# ============================================================
def predict(model, features):
    if model is None:
        return None

    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    try:
        out = model.predict(X)
        return out[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# ============================================================
# FILE READER
# ============================================================
def read_molecule_file(f):
    name = f.name
    data = f.getvalue().decode("utf-8", errors="ignore")

    if name.endswith(".smi"):
        return data.split()[0]

    if name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(data)
        return Chem.MolToSmiles(mol) if mol else None

    if name.endswith(".sdf"):
        block = data.split("$$$$")[0]
        mol = Chem.MolFromMolBlock(block)
        return Chem.MolToSmiles(mol) if mol else None

    return None

# ============================================================
# DATASET FOR FRET
# ============================================================
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except:
        return None

# ============================================================
# UI TABS
# ============================================================
st.title("FluroML: Molecular Fluorescence Predictor (Final 2026)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Prediction",
    "Emission Prediction",
    "FRET Analysis"
])

# ============================================================
# TAB 1 — CLASSIFICATION
# ============================================================
with tab1:
    st.subheader("🧪 Fluorescence Classification")

    method = st.radio("Input:", ["SMILES", "Draw", "Upload"], key="clf_meth")

    smiles = ""
    if method == "SMILES":
        smiles = st.text_input("Enter SMILES", key="clf_smi")
    elif method == "Upload":
        up = st.file_uploader("Upload molecule", type=["smi", "mol", "sdf"])
        if up:
            smiles = read_molecule_file(up)
    else:
        smiles = st_ketcher("")

    if smiles:
        img = safe_mol_image(smiles)
        if img:
            st.image(img)

        feat = smiles_to_morgan(smiles)
        if feat is not None:
            pred = predict(model_fluorescence, feat)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# ============================================================
# TAB 2 — ABSORPTION
# ============================================================
with tab2:
    st.subheader("🌈 Absorption Max Prediction")

    method = st.radio("Input:", ["SMILES", "Draw", "Upload"], key="abs_meth")
    smi = ""

    if method == "SMILES":
        smi = st.text_input("Enter SMILES", key="abs_smi")
    elif method == "Upload":
        up = st.file_uploader("Upload molecule", type=["smi","mol","sdf"])
        if up:
            smi = read_molecule_file(up)
    else:
        smi = st_ketcher("")

    solvent = st.text_input("Solvent SMILES", "O", key="abs_sol")

    if smi and solvent:
        img = safe_mol_image(smi)
        if img:
            st.image(img)

        d1 = smiles_to_descriptors(smi)
        d2 = smiles_to_descriptors(solvent)

        if d1 is not None and d2 is not None:
            feat = pd.concat([d1, d2], axis=1)
            pred = predict(model_regression, feat)
            if pred is not None:
                st.success(f"Predicted Absorption: {pred:.2f} nm")

# ============================================================
# TAB 3 — EMISSION
# ============================================================
with tab3:
    st.subheader("🔦 Emission Max Prediction")

    method = st.radio("Input:", ["SMILES", "Draw", "Upload"], key="emi_meth")
    smi = ""

    if method == "SMILES":
        smi = st.text_input("Enter SMILES", key="emi_smi")
    elif method == "Upload":
        up = st.file_uploader("Upload molecule", type=["smi","mol","sdf"])
        if up:
            smi = read_molecule_file(up)
    else:
        smi = st_ketcher("")

    solvent = st.text_input("Solvent SMILES", "O", key="emi_sol")

    if smi and solvent:
        img = safe_mol_image(smi)
        if img:
            st.image(img)

        d1 = smiles_to_descriptors(smi)
        d2 = smiles_to_descriptors(solvent)

        if d1 is not None and d2 is not None:
            feat = pd.concat([d1, d2], axis=1)
            pred = predict(model_emission, feat)
            if pred is not None:
                st.success(f"Predicted Emission: {pred:.2f} nm")

# ============================================================
# TAB 4 — ADVANCED FRET ANALYSIS
# ============================================================
with tab4:
    st.subheader("🔬 FRET Pair Analysis")
    st.info("Provide donor or acceptor SMILES to find the best spectral match.")

    # Donor and Acceptor Inputs
    donor = st.text_input("Donor SMILES", key="donor_smi")
    acceptor = st.text_input("Acceptor SMILES", key="acc_smi")

    df = load_dataset()

    if donor:
        d1 = smiles_to_descriptors(donor)
        d2 = smiles_to_descriptors("O")
        if d1 is not None:
            pred_donor_em = predict(model_emission, pd.concat([d1,d2],axis=1))
            st.success(f"Predicted Donor Emission: {pred_donor_em:.2f} nm")

    if acceptor:
        a1 = smiles_to_descriptors(acceptor)
        a2 = smiles_to_descriptors("O")
        if a1 is not None:
            pred_acc_abs = predict(model_regression, pd.concat([a1,a2],axis=1))
            st.success(f"Predicted Acceptor Absorption: {pred_acc_abs:.2f} nm")

    st.markdown("---")

# Footer
st.caption("FluroML © P. Deshmukh (2026)")
