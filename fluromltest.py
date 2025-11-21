import os
import streamlit as st
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
import joblib
import pandas as pd
import deepchem as dc
from PIL import Image

# Ketcher integration
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("❗ Please install streamlit-ketcher to enable molecule drawing.")
        return ""

# ---------------- PAGE SETUP ----------------
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.title("FluroML : Molecular Fluorescence Predictor")

# ---------------- LOGOS ----------------
def load_logo(path):
    return Image.open(path) if os.path.exists(path) else None

logo_classification = load_logo("logo_classification.png")
logo_spectra = load_logo("logo_spectra.png")
logo_fret = load_logo("logo_fret.png")

# ---------------- MODELS ----------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except:
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression = load_model("new_best_regressor_compatible.joblib")
model_emission = load_model("best_regressor_emission_compatible.joblib")

# ---------------- SAFE SVG MOLECULE DRAW ----------------
def draw_molecule_svg(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    drawer = rdMolDraw2D.MolDraw2DSVG(300, 300)
    rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    return drawer.GetDrawingText().replace("svg:", "")

# ---------------- DESCRIPTORS ----------------
def smiles_to_morgan(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    return featurizer.featurize([mol])[0]

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    return pd.DataFrame(featurizer.featurize([mol]))

def predict(model, features):
    import numpy as np
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return model.predict(X)[0]

# ---------------- FILE LOADER ----------------
def read_molecule_file(file):
    content = file.getvalue().decode("utf-8", errors="ignore")
    if file.name.endswith(".smi"):
        return content.strip().split()[0]
    elif file.name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(content)
        return Chem.MolToSmiles(mol) if mol else None
    elif file.name.endswith(".sdf"):
        return Chem.MolToSmiles(Chem.MolFromMolBlock(content.split("$$$$")[0]))

# ---------------- DATASET ----------------
@st.cache_data
def load_dataset():
    return pd.read_csv("All Properties with Finguprints_3.csv")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ================= TAB 1 =================
with tab1:
    col1, col2 = st.columns([1,4])
    if logo_classification:
        col1.image(logo_classification)
    col2.header("Fluorescence Classification")

    method = st.radio("Input Method:", ("SMILES", "Draw", "Upload"))
    smiles = ""

    if method == "Draw":
        smiles = st_ketcher("")
    elif method == "Upload":
        file = st.file_uploader("Upload file", type=["smi","mol","sdf"])
        if file:
            smiles = read_molecule_file(file)
    else:
        smiles = st.text_input("Enter SMILES")

    if smiles:
        svg = draw_molecule_svg(smiles)
        if svg:
            st.markdown(svg, unsafe_allow_html=True)

        pred = predict(model_fluorescence, smiles_to_morgan(smiles))
        st.success("Fluorescent ✅" if int(pred) == 1 else "Non-Fluorescent ❌")

# ================= TAB 2 =================
with tab2:
    col1, col2 = st.columns([1,4])
    if logo_spectra:
        col1.image(logo_spectra)
    col2.header("Absorption Prediction")

    abs_smiles = st_ketcher("")
    solvent = st.text_input("Solvent SMILES", value="O")

    if abs_smiles:
        svg = draw_molecule_svg(abs_smiles)
        if svg:
            st.markdown(svg, unsafe_allow_html=True)

        mol_desc = smiles_to_descriptors(abs_smiles)
        solv_desc = smiles_to_descriptors(solvent)
        features = pd.concat([mol_desc, solv_desc], axis=1)
        pred = predict(model_regression, features)
        st.success(f"Predicted Absorption: {pred:.2f} nm")

# ================= TAB 3 =================
with tab3:
    col1, col2 = st.columns([1,4])
    if logo_spectra:
        col1.image(logo_spectra)
    col2.header("Emission Prediction")

    em_smiles = st_ketcher("")
    solvent = st.text_input("Solvent SMILES", value="O")

    if em_smiles:
        svg = draw_molecule_svg(em_smiles)
        if svg:
            st.markdown(svg, unsafe_allow_html=True)

        mol_desc = smiles_to_descriptors(em_smiles)
        solv_desc = smiles_to_descriptors(solvent)
        features = pd.concat([mol_desc, solv_desc], axis=1)
        pred = predict(model_emission, features)
        st.success(f"Predicted Emission: {pred:.2f} nm")

# ================= TAB 4 =================
with tab4:
    col1, col2 = st.columns([1,4])
    if logo_fret:
        col1.image(logo_fret)
    col2.header("FRET Analysis")

    donor_smiles = st_ketcher("")

    if donor_smiles:
        df = load_dataset()
        donor_desc = smiles_to_descriptors(donor_smiles)
        solvent_desc = smiles_to_descriptors("O")
        features = pd.concat([donor_desc, solvent_desc], axis=1)
        donor_em = predict(model_emission, features)

        df = df[df['Fluorescent labeling'].str.lower() == "yes"].copy()
        df["Δ"] = abs(df["AbsorptioMax (nm)"] - donor_em)
        top5 = df.sort_values("Δ").head(5)

        st.dataframe(top5[["Smiles","AbsorptioMax (nm)","EmissionMax (nm)","Δ"]])

# Footer
st.write("---")
st.caption("FluroML © PDeshmukh")
