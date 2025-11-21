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

# ----------------- Streamlit config -----------------
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("FluroML: Molecular Fluorescence Predictor")

# ----------------- Model loading -----------------
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

# ----------------- Feature helpers -----------------
def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    return dc.feat.CircularFingerprint(radius=3, size=1024).featurize([mol])[0]

def smiles_to_descriptors(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    return pd.DataFrame(dc.feat.MACCSKeysFingerprint().featurize([mol]))

def predict(model, features):
    import numpy as np
    if model is None or features is None:
        return None
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return model.predict(X)[0]

# ----------------- File reader -----------------
def read_molecule_file(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    name = uploaded_file.name.lower()

    if name.endswith(".smi"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            st.error("SMI file is empty.")
            return None
        return lines[0].split()[0]

    if name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(text)
        return Chem.MolToSmiles(mol) if mol else None

    if name.endswith(".sdf"):
        blocks = [blk for blk in text.split("$$$$") if blk.strip()]
        if not blocks:
            st.error("No molecules found in SDF.")
            return None
        mol = Chem.MolFromMolBlock(blocks[0])
        return Chem.MolToSmiles(mol) if mol else None

    st.error("Unsupported file format.")
    return None

# ----------------- Dataset for FRET -----------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ----------------- Tabs -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# =====================================================
# TAB 1 – Fluorescence Classification
# =====================================================
with tab1:
    st.header("🔎 Fluorescence Classification")

    method = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="clf_input_method"
    )
    smiles = ""

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES", key="clf_smiles")
    elif method == "Upload File":
        file = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="clf_file"
        )
        if file:
            smiles = read_molecule_file(file) or ""
    else:
        smiles = st_ketcher("")
        if smiles:
            st.write(f"SMILES from drawing: `{smiles}`")

    if smiles and model_fluorescence:
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            pred = predict(model_fluorescence, feats)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# =====================================================
# TAB 2 – Absorption Max Prediction
# =====================================================
with tab2:
    st.header("📈 Absorption Max Prediction")

    method2 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="abs_input_method"
    )
    abs_smiles = ""
    solvent = st.text_input("Solvent SMILES", value="O", key="abs_solvent")

    if method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter SMILES", key="abs_smiles")
    elif method2 == "Upload File":
        file2 = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="abs_file"
        )
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        abs_smiles = st_ketcher("")
        if abs_smiles:
            st.write(f"SMILES from drawing: `{abs_smiles}`")

    if abs_smiles and solvent and model_regression:
        f1 = smiles_to_descriptors(abs_smiles)
        f2 = smiles_to_descriptors(solvent)
        if f1 is not None and f2 is not None:
            feats = pd.concat([f1, f2], axis=1)
            pred = predict(model_regression, feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# =====================================================
# TAB 3 – Emission Max Prediction
# =====================================================
with tab3:
    st.header("📈 Emission Max Prediction")

    method3 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="emi_input_method"
    )
    em_smiles = ""
    solvent_em = st.text_input("Solvent SMILES", value="O", key="emi_solvent")

    if method3 == "SMILES Input":
        em_smiles = st.text_input("Enter SMILES", key="emi_smiles")
    elif method3 == "Upload File":
        file3 = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="emi_file"
        )
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        em_smiles = st_ketcher("")
        if em_smiles:
            st.write(f"SMILES from drawing: `{em_smiles}`")

    if em_smiles and solvent_em and model_emission:
        f1 = smiles_to_descriptors(em_smiles)
        f2 = smiles_to_descriptors(solvent_em)
        if f1 is not None and f2 is not None:
            feats = pd.concat([f1, f2], axis=1)
            pred = predict(model_emission, feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {pred:.2f} nm")

# =====================================================
# TAB 4 – FRET Analysis
# =====================================================
with tab4:
    st.header("🔗 FRET Pair Analysis")

    method4 = st.radio(
        "Donor Input:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="fret_input_method"
    )
    donor_smiles = ""

    if method4 == "SMILES Input":
        donor_smiles = st.text_input("Donor SMILES", key="fret_donor_smiles")
    elif method4 == "Upload File":
        file4 = st.file_uploader(
            "Upload donor", type=["smi", "mol", "sdf"], key="fret_file"
        )
        if file4:
            donor_smiles = read_molecule_file(file4) or ""
    else:
        donor_smiles = st_ketcher("")
        if donor_smiles:
            st.write(f"SMILES from drawing: `{donor_smiles}`")

    if donor_smiles and model_emission:
        df = load_dataset()
        if df is not None and "AbsorptioMax (nm)" in df.columns:
            f1 = smiles_to_descriptors(donor_smiles)
            f2 = smiles_to_descriptors("O")
            if f1 is not None and f2 is not None:
                feats = pd.concat([f1, f2], axis=1)
                donor_em = predict(model_emission, feats)
                if donor_em is not None:
                    df = df.copy()
                    df["Δ (nm)"] = (df["AbsorptioMax (nm)"] - donor_em).abs()
                    top5 = df.sort_values("Δ (nm)").head(5)
                    st.write(f"Predicted donor emission: {donor_em:.2f} nm")
                    st.markdown("**Top 5 closest FRET partners (by absorption match):**")
                    st.table(top5[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ (nm)"]])
        else:
            st.error("Dataset missing 'AbsorptioMax (nm)' column.")

# ----------------- Footer -----------------
st.write("---")
st.caption("FluroML-©PDeshmukh")
