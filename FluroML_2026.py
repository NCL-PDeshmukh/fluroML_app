# FluroML_2026_FINAL.py
# Final Streamlit app — cloud-safe 2D rendering + model feature shapes matched.

import warnings
warnings.filterwarnings("ignore")  # silence sklearn / deepchem noisy warnings

import streamlit as st
import pandas as pd
import numpy as np
import joblib

# RDKit & Draw (MolToImage)
from rdkit import Chem
from rdkit.Chem import Draw

# DeepChem featurizers
import deepchem as dc

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="FluroML: Molecular Fluorescence Predictor", layout="wide")
st.title("FluroML: Molecular Fluorescence Predictor (FINAL)")

# -------------------------------
# Cached model loader
# -------------------------------
@st.cache_resource
def load_joblib_model(path):
    try:
        m = joblib.load(path)
        return m
    except Exception as e:
        st.error(f"Failed to load model {path}: {e}")
        return None

# Paths: adjust if your models are located elsewhere
model_fluoro = load_joblib_model("best_classifier_compatible.joblib")   # expects 1024
model_abs     = load_joblib_model("new_best_regressor_compatible.joblib")# expects 332
model_emit    = load_joblib_model("best_regressor_emission_compatible.joblib") # expects 332

# -------------------------------
# Feature generators (DeepChem)
# -------------------------------
# Morgan fingerprint (1024) — used by fluorescence model
_morgan = dc.feat.CircularFingerprint(radius=3, size=1024)
# MACCS keys (166) — used by absorption/emission models (molecule + solvent)
_maccs = dc.feat.MACCSKeysFingerprint()

def compute_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = _morgan.featurize([mol])[0]
        return np.array(arr, dtype=float)
    except Exception as e:
        st.error(f"Morgan featurization error: {e}")
        return None

def compute_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = _maccs.featurize([mol])[0]
        return np.array(arr, dtype=float)
    except Exception as e:
        st.error(f"MACCS featurization error: {e}")
        return None

def build_features_fluoro(smiles: str):
    f = compute_morgan(smiles)
    return f.reshape(1, -1) if f is not None else None

def build_features_abs_em(smiles: str, solvent_smiles: str):
    m = compute_maccs(smiles)
    s = compute_maccs(solvent_smiles)
    if m is None or s is None:
        return None
    return np.concatenate([m, s]).reshape(1, -1)

# -------------------------------
# 2D renderer — guaranteed cloud-safe
# -------------------------------
def render_molecule_image(smiles: str, size=(320, 320)):
    """Render molecule as a PNG using RDKit -> PIL (MolToImage). Always works on Streamlit Cloud."""
    if not smiles:
        st.warning("No SMILES provided.")
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string — RDKit cannot parse it.")
        return
    try:
        # compute 2D coords for nicer layout
        try:
            Chem.rdDepictor.Compute2DCoords(mol)
        except Exception:
            pass
        img = Draw.MolToImage(mol, size=size)  # PIL image
        st.image(img, caption="2D Structure", use_column_width=False)
    except Exception as e:
        st.error(f"Failed to render structure: {e}")

# -------------------------------
# CSV dataset loader for FRET (optional)
# -------------------------------
@st.cache_data
def load_fret_dataset(path="All Properties with Finguprints_3.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

# -------------------------------
# UI layout: tabs
# -------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -------------------------------
# TAB 1: Fluorescence Classification (Morgan 1024)
# -------------------------------
with tab1:
    st.header("🧪 Fluorescence Classification (Morgan 1024)")
    method = st.radio("Input method:", ["SMILES input", "Upload file"], key="t1_method")
    smiles_t1 = ""

    if method == "SMILES input":
        smiles_t1 = st.text_input("Enter SMILES:", key="t1_smiles")
    else:
        up = st.file_uploader("Upload (.smi/.mol/.sdf)", type=["smi", "mol", "sdf"], key="t1_file")
        if up:
            # safe file reading
            data = up.getvalue()
            try:
                content = data.decode("utf-8", errors="ignore")
                if up.name.lower().endswith(".smi"):
                    line = next((ln for ln in content.splitlines() if ln.strip()), "")
                    smiles_t1 = line.split()[0] if line else ""
                elif up.name.lower().endswith(".mol"):
                    m = Chem.MolFromMolBlock(content)
                    smiles_t1 = Chem.MolToSmiles(m) if m else ""
                elif up.name.lower().endswith(".sdf"):
                    blocks = content.split("$$$$")
                    m = Chem.MolFromMolBlock(blocks[0])
                    smiles_t1 = Chem.MolToSmiles(m) if m else ""
            except Exception:
                st.error("Failed to read uploaded file.")

    if smiles_t1:
        render_molecule_image(smiles_t1, size=(320,320))
        feats = build_features_fluoro(smiles_t1)
        if feats is None:
            st.error("Feature generation failed.")
        else:
            if model_fluoro is None:
                st.error("Fluorescence model not loaded.")
            else:
                try:
                    pred = model_fluoro.predict(feats)[0]
                    st.success("Predicted: Fluorescent" if int(pred) == 1 else "Predicted: Non-Fluorescent")
                    st.write(f"Used feature shape: {feats.shape}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# -------------------------------
# TAB 2: Absorption Max Prediction (MACCS mol + MACCS solvent = 332)
# -------------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction (MACCS mol + MACCS solvent → 332 features)")
    method2 = st.radio("Input method:", ["SMILES input", "Upload file"], key="t2_method")
    mol_smiles2 = ""
    if method2 == "SMILES input":
        mol_smiles2 = st.text_input("Enter molecule SMILES:", key="t2_smiles")
    else:
        up = st.file_uploader("Upload molecule (.smi/.mol/.sdf)", type=["smi","mol","sdf"], key="t2_file")
        if up:
            data = up.getvalue()
            try:
                content = data.decode("utf-8", errors="ignore")
                if up.name.lower().endswith(".smi"):
                    line = next((ln for ln in content.splitlines() if ln.strip()), "")
                    mol_smiles2 = line.split()[0] if line else ""
                elif up.name.lower().endswith(".mol"):
                    m = Chem.MolFromMolBlock(content)
                    mol_smiles2 = Chem.MolToSmiles(m) if m else ""
                elif up.name.lower().endswith(".sdf"):
                    blocks = content.split("$$$$")
                    m = Chem.MolFromMolBlock(blocks[0])
                    mol_smiles2 = Chem.MolToSmiles(m) if m else ""
            except Exception:
                st.error("Failed to read uploaded file.")

    solvent2 = st.text_input("Solvent SMILES (default O):", value="O", key="t2_solvent")

    if mol_smiles2:
        render_molecule_image(mol_smiles2, size=(320,320))

    if mol_smiles2 and solvent2:
        feats2 = build_features_abs_em(mol_smiles2, solvent2)
        if feats2 is None:
            st.error("Failed to build absorption features.")
        else:
            if model_abs is None:
                st.error("Absorption model not loaded.")
            else:
                try:
                    pred2 = model_abs.predict(feats2)[0]
                    st.success(f"Predicted Absorption Max: {float(pred2):.2f} nm")
                    st.write(f"Used feature shape: {feats2.shape}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# -------------------------------
# TAB 3: Emission Max Prediction (MACCS mol + MACCS solvent = 332)
# -------------------------------
with tab3:
    st.header("🔦 Emission Max Prediction (MACCS mol + MACCS solvent → 332 features)")
    method3 = st.radio("Input method:", ["SMILES input", "Upload file"], key="t3_method")
    mol_smiles3 = ""
    if method3 == "SMILES input":
        mol_smiles3 = st.text_input("Enter molecule SMILES:", key="t3_smiles")
    else:
        up = st.file_uploader("Upload molecule (.smi/.mol/.sdf)", type=["smi","mol","sdf"], key="t3_file")
        if up:
            data = up.getvalue()
            try:
                content = data.decode("utf-8", errors="ignore")
                if up.name.lower().endswith(".smi"):
                    line = next((ln for ln in content.splitlines() if ln.strip()), "")
                    mol_smiles3 = line.split()[0] if line else ""
                elif up.name.lower().endswith(".mol"):
                    m = Chem.MolFromMolBlock(content)
                    mol_smiles3 = Chem.MolToSmiles(m) if m else ""
                elif up.name.lower().endswith(".sdf"):
                    blocks = content.split("$$$$")
                    m = Chem.MolFromMolBlock(blocks[0])
                    mol_smiles3 = Chem.MolToSmiles(m) if m else ""
            except Exception:
                st.error("Failed to read uploaded file.")

    solvent3 = st.text_input("Solvent SMILES (default O):", value="O", key="t3_solvent")

    if mol_smiles3:
        render_molecule_image(mol_smiles3, size=(320,320))

    if mol_smiles3 and solvent3:
        feats3 = build_features_abs_em(mol_smiles3, solvent3)
        if feats3 is None:
            st.error("Failed to build emission features.")
        else:
            if model_emit is None:
                st.error("Emission model not loaded.")
            else:
                try:
                    pred3 = model_emit.predict(feats3)[0]
                    st.success(f"Predicted Emission Max: {float(pred3):.2f} nm")
                    st.write(f"Used feature shape: {feats3.shape}")
                except Exception as e:
                    st.error(f"Prediction error: {e}")

# -------------------------------
# TAB 4: FRET Analysis (dataset-based partner search)
# -------------------------------
with tab4:
    st.header("🔬 FRET Analysis")
    col1, col2 = st.columns(2)

    donor_smiles = ""
    acceptor_smiles = ""

    with col1:
        st.subheader("Donor")
        donor_smiles = st.text_input("Donor SMILES:", key="donor_smi")
    with col2:
        st.subheader("Acceptor")
        acceptor_smiles = st.text_input("Acceptor SMILES:", key="acceptor_smi")

    if donor_smiles:
        st.write("Donor structure:")
        render_molecule_image(donor_smiles, size=(300,300))

    if acceptor_smiles:
        st.write("Acceptor structure:")
        render_molecule_image(acceptor_smiles, size=(300,300))

    df_fret = load_fret_dataset()
    if (donor_smiles or acceptor_smiles) and df_fret is None:
        st.warning("FRET dataset CSV not found — dataset-based partner search unavailable.")

    # If donor provided -> predict donor emission and list top acceptors from dataset
    if donor_smiles and model_emit is not None and df_fret is not None:
        feats_d = build_features_abs_em(donor_smiles, "O")
        if feats_d is not None:
            try:
                donor_em = model_emit.predict(feats_d)[0]
                st.info(f"Predicted Donor Emission (nm): {donor_em:.1f}")
                dfc = df_fret[df_fret['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                dfc['Δ'] = (dfc['AbsorptioMax (nm)'] - donor_em).abs()
                st.subheader("Top 5 acceptors by Δ (abs - donor_em)")
                st.table(dfc.sort_values('Δ').head(5)[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])
            except Exception as e:
                st.error(f"Error predicting donor emission: {e}")

    # If acceptor provided -> predict acceptor absorption and list top donors from dataset
    if acceptor_smiles and model_abs is not None and df_fret is not None:
        feats_a = build_features_abs_em(acceptor_smiles, "O")
        if feats_a is not None:
            try:
                acc_abs = model_abs.predict(feats_a)[0]
                st.info(f"Predicted Acceptor Absorption (nm): {acc_abs:.1f}")
                dfc = df_fret[df_fret['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                dfc['Δ'] = (dfc['EmissionMax (nm)'] - acc_abs).abs()
                st.subheader("Top 5 donors by Δ (donor_em - acc_abs)")
                st.table(dfc.sort_values('Δ').head(5)[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])
            except Exception as e:
                st.error(f"Error predicting acceptor absorption: {e}")

# Footer
st.write("---")
st.caption("FluroML ©PDeshmukh")
