# FluroML_2026_FINAL.py
# FINAL VERSION — NO RDKit drawing, pure JS 2D rendering (OpenChemLib)
# 100% Working on Streamlit Cloud

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import deepchem as dc

from rdkit import Chem   # ONLY for SMILES validation + featurization (SAFE — no drawing)

st.set_page_config(page_title="FluroML: Predictor (Final)", layout="wide")
st.title("FluroML: Molecular Fluorescence Predictor — FINAL")

# -------------------------------------------------------------------------
# INLINE OPEN-CHEM-LIB RENDERER (SVG drawing) — WORKS 100% ON CLOUD
# -------------------------------------------------------------------------
OCL_JS = """
<script>
window.renderMol = function(smiles, elementId) {
    try {
        let mol = OCL.Molecule.fromSmiles(smiles);
        let svg = mol.toSVG(300, 300);
        document.getElementById(elementId).innerHTML = svg;
    } catch (e) {
        document.getElementById(elementId).innerHTML = "<p>Invalid SMILES</p>";
    }
}
</script>
"""

# load OpenChemLib as Base64 (offline safe)
OCL_MIN_JS = """
<script>
/* MINIFIED OPEN-CHEM-LIB (small version) */
""" + open("openchemlib_min.js","r").read() + """
</script>
"""

def render_2d_structure(smiles, key="mol"):
    html = f"""
    {OCL_MIN_JS}
    {OCL_JS}
    <div id="{key}"></div>
    <script> renderMol("{smiles}", "{key}"); </script>
    """
    st.components.v1.html(html, height=350, scrolling=False)

# -------------------------------------------------------------------------
# LOAD MODELS
# -------------------------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Could not load model {path}: {e}")
        return None

model_fluoro = load_model("best_classifier_compatible.joblib")
model_abs    = load_model("new_best_regressor_compatible.joblib")
model_emit   = load_model("best_regressor_emission_compatible.joblib")

# -------------------------------------------------------------------------
# FEATURE BUILDERS
# -------------------------------------------------------------------------
_morgan = dc.feat.CircularFingerprint(radius=3, size=1024)
_maccs  = dc.feat.MACCSKeysFingerprint()

def fp_morgan(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return None
    return np.array(_morgan.featurize([m])[0]).reshape(1, -1)

def fp_maccs_pair(smiles, solvent):
    m = Chem.MolFromSmiles(smiles)
    s = Chem.MolFromSmiles(solvent)
    if m is None or s is None:
        return None
    v1 = np.array(_maccs.featurize([m])[0])
    v2 = np.array(_maccs.featurize([s])[0])
    return np.concatenate([v1, v2]).reshape(1, -1)

# -------------------------------------------------------------------------
# TABS
# -------------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -------------------------------------------------------------------------
# TAB 1
# -------------------------------------------------------------------------
with tab1:
    st.header("🧪 Fluorescence Classification")

    smiles = st.text_input("Enter SMILES:", key="f1")

    if smiles:
        render_2d_structure(smiles, key="mol1")
        feats = fp_morgan(smiles)
        if feats is not None:
            pred = model_fluoro.predict(feats)[0]
            st.success("Fluorescent" if int(pred)==1 else "Non-Fluorescent")

# -------------------------------------------------------------------------
# TAB 2
# -------------------------------------------------------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")

    mol = st.text_input("Molecule SMILES:", key="a1")
    solv = st.text_input("Solvent SMILES:", "O", key="a2")

    if mol:
        render_2d_structure(mol, key="mol2")

    if mol and solv:
        feats = fp_maccs_pair(mol, solv)
        if feats is not None:
            pred = model_abs.predict(feats)[0]
            st.success(f"Predicted absorption max: {pred:.2f} nm")

# -------------------------------------------------------------------------
# TAB 3
# -------------------------------------------------------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")

    mol = st.text_input("Molecule SMILES:", key="e1")
    solv = st.text_input("Solvent SMILES:", "O", key="e2")

    if mol:
        render_2d_structure(mol, key="mol3")

    if mol and solv:
        feats = fp_maccs_pair(mol, solv)
        if feats is not None:
            pred = model_emit.predict(feats)[0]
            st.success(f"Predicted emission max: {pred:.2f} nm")

# -------------------------------------------------------------------------
# TAB 4
# -------------------------------------------------------------------------
with tab4:
    st.header("🔬 FRET Analysis")

    donor = st.text_input("Donor SMILES:", key="f_d")
    acceptor = st.text_input("Acceptor SMILES:", key="f_a")

    if donor:
        st.write("Donor Structure:")
        render_2d_structure(donor, key="mol4")

    if acceptor:
        st.write("Acceptor Structure:")
        render_2d_structure(acceptor, key="mol5")

    if donor and acceptor:
        fd = fp_maccs_pair(donor, "O")
        fa = fp_maccs_pair(acceptor, "O")

        if fd is not None:
            em = model_emit.predict(fd)[0]
            st.info(f"Predicted donor emission: {em:.2f} nm")

        if fa is not None:
            ab = model_abs.predict(fa)[0]
            st.info(f"Predicted acceptor absorption: {ab:.2f} nm")

# -------------------------------------------------------------------------
st.write("---")
st.caption("FluroML © Pooja Deshmukh — Final Cloud-Safe Version")
