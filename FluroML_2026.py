import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
import joblib
import deepchem as dc
import urllib
import streamlit.components.v1 as components

# -------------------------------------------------------
# Page Setup
# -------------------------------------------------------
st.set_page_config(
    page_title="FluroML: Molecular Fluorescence Predictor",
    layout="wide",
)

# -------------------------------------------------------------------
# SAFE 2D STRUCTURE RENDERING — RDKit SVG (WORKS ON STREAMLIT CLOUD)
# -------------------------------------------------------------------
def show_mol(smiles: str, width=450):
    """Render a molecule using RDKit SVG (cloud-safe)."""
    if not smiles:
        return
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return
    svg = Draw.MolToSVG(mol, size=(width, width))
    svg_clean = svg.replace("<?xml version='1.0' encoding='UTF-8'?>", "")
    st.write(svg_clean, unsafe_allow_html=True)

# -------------------------------------------------------
# Enhanced molecular representation (Morgan + MACCS)
# -------------------------------------------------------
def smiles_to_features(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None

    # Morgan fingerprint (1024 bits)
    fp_featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    fp = fp_featurizer.featurize([mol])[0]

    # MACCS (166 bits)
    desc_featurizer = dc.feat.MACCSKeysFingerprint()
    desc = desc_featurizer.featurize([mol])[0]

    # Concatenate
    features = np.concatenate([fp, desc])
    return features  # shape = (1190,)

# -------------------------------------------------------
# Descriptor only (for solvent-based models)
# -------------------------------------------------------
def smiles_to_descriptors_df(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    X = featurizer.featurize([mol])
    return pd.DataFrame(X)

# -------------------------------------------------------
# File Reader
# -------------------------------------------------------
def read_molecule_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue().decode("utf-8", errors="ignore")

    if name.endswith(".smi"):
        smi = data.splitlines()[0].split()[0]
        return smi

    elif name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(data)
        if mol:
            return Chem.MolToSmiles(mol)

    elif name.endswith(".sdf"):
        blocks = data.split("$$$$")
        mol = Chem.MolFromMolBlock(blocks[0])
        if mol:
            return Chem.MolToSmiles(mol)

    st.error("Invalid molecule file.")
    return None

# -------------------------------------------------------
# Load Models
# -------------------------------------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading {path}: {e}")
        return None

model_fluoro = load_model("best_classifier_compatible.joblib")
model_abs = load_model("new_best_regressor_compatible.joblib")
model_em = load_model("best_regressor_emission_compatible.joblib")

# -------------------------------------------------------
# Prediction Helper
# -------------------------------------------------------
def predict(model, X):
    if model is None:
        return None
    if isinstance(X, pd.DataFrame):
        X = X.values
    X = np.array(X).reshape(1, -1)
    return model.predict(X)[0]

# -------------------------------------------------------
# Load dataset for FRET
# -------------------------------------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except:
        return None

# -------------------------------------------------------
# App Title
# -------------------------------------------------------
st.title("FluroML: Molecular Fluorescence Predictor (2D Structures)")

tabs = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -------------------------------------------------------
# TAB 1 — Fluorescence Classification
# -------------------------------------------------------
with tabs[0]:
    st.header("🧪 Fluorescence Classification")

    method = st.radio("Input Method:", 
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="fluoro_input_method")

    smiles = ""

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="fluoro_smiles")

    elif method == "Upload File":
        file = st.file_uploader("Upload molecule file:", type=["smi","mol","sdf"], key="fluoro_file")
        if file:
            smiles = read_molecule_file(file)

    else:
        html = """
        <script src="https://unpkg.com/jsme-editor"></script>
        <div id="jsme" style="width:400px; height:350px;"></div>
        <script>
            var jsme = new JSApplet.JSME("jsme", "400px", "350px", {"options":"query"});
            function sendSmiles(){
                const value = jsme.smiles();
                window.parent.postMessage({jsme_smiles:value}, "*");
            }
            setInterval(sendSmiles, 800);
        </script>
        """
        components.html(html, height=380)
        js = st._get_script_run_ctx().session_state.get("jsme_smiles", "")
        if js:
            smiles = js

    if smiles:
        st.subheader("2D Structure")
        show_mol(smiles)

        X = smiles_to_features(smiles)
        if X is not None:
            pred = predict(model_fluoro, X)
            if pred == 1:
                st.success("Fluorescent")
            else:
                st.error("Non-Fluorescent")

# -------------------------------------------------------
# TAB 2 — Absorption Max Prediction
# -------------------------------------------------------
with tabs[1]:
    st.header("🌈 Absorption Max Prediction")

    method = st.radio("Input Method:", 
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="abs_input_method")

    abs_smiles = ""

    if method == "SMILES Input":
        abs_smiles = st.text_input("Enter molecule SMILES:", key="abs_smiles")

    elif method == "Upload File":
        file = st.file_uploader("Upload molecule:", type=["smi","mol","sdf"], key="abs_file")
        if file:
            abs_smiles = read_molecule_file(file)

    else:
        html = """
        <script src="https://unpkg.com/jsme-editor"></script>
        <div id="jsme_abs" style="width:400px; height:350px;"></div>
        <script>
            var jsme_abs = new JSApplet.JSME("jsme_abs", "400px", "350px", {"options":"query"});
            function sendAbs(){
                const value = jsme_abs.smiles();
                window.parent.postMessage({abs_jsme_smiles:value}, "*");
            }
            setInterval(sendAbs, 800);
        </script>
        """
        components.html(html, height=380)
        js = st._get_script_run_ctx().session_state.get("abs_jsme_smiles", "")
        if js:
            abs_smiles = js

    solvent = st.text_input("Solvent SMILES:", key="abs_solvent")

    if abs_smiles and solvent:
        st.subheader("2D Structure")
        show_mol(abs_smiles)

        d1 = smiles_to_descriptors_df(abs_smiles)
        d2 = smiles_to_descriptors_df(solvent)

        if d1 is not None and d2 is not None:
            X = pd.concat([d1, d2], axis=1)
            pred = predict(model_abs, X)
            st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# -------------------------------------------------------
# TAB 3 — Emission Max Prediction
# -------------------------------------------------------
with tabs[2]:
    st.header("🔦 Emission Max Prediction")

    method = st.radio("Input Method:", 
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="em_input_method")

    em_smiles = ""

    if method == "SMILES Input":
        em_smiles = st.text_input("Enter SMILES:", key="em_smiles")

    elif method == "Upload File":
        file = st.file_uploader("Upload file:", type=["smi","mol","sdf"], key="em_file")
        if file:
            em_smiles = read_molecule_file(file)

    else:
        html = """
        <script src="https://unpkg.com/jsme-editor"></script>
        <div id="jsme_em" style="width:400px; height:350px;"></div>
        <script>
            var jsme_em = new JSApplet.JSME("jsme_em", "400px", "350px", {"options":"query"});
            function sendEm(){
                const value = jsme_em.smiles();
                window.parent.postMessage({em_jsme_smiles:value}, "*");
            }
            setInterval(sendEm, 800);
        </script>
        """
        components.html(html, height=380)
        js = st._get_script_run_ctx().session_state.get("em_jsme_smiles", "")
        if js:
            em_smiles = js

    solvent_em = st.text_input("Solvent SMILES:", key="em_solvent")

    if em_smiles and solvent_em:
        st.subheader("2D Structure")
        show_mol(em_smiles)

        d1 = smiles_to_descriptors_df(em_smiles)
        d2 = smiles_to_descriptors_df(solvent_em)

        if d1 is not None and d2 is not None:
            X = pd.concat([d1, d2], axis=1)
            pred = predict(model_em, X)
            st.success(f"Predicted Emission Max: {pred:.2f} nm")

# -------------------------------------------------------
# TAB 4 — FRET Analysis
# -------------------------------------------------------
with tabs[3]:
    st.header("🔬 FRET Analysis")

    col1, col2 = st.columns(2)

    # -------- Donor -------
    with col1:
        st.subheader("Donor Molecule")
        method = st.radio("Input Method:", 
                          ["SMILES", "Draw (JSME)", "Upload"], key="donor_method")

        donor = ""
        if method == "SMILES":
            donor = st.text_input("Donor SMILES:", key="donor_smi")
        elif method == "Upload":
            file = st.file_uploader("Upload Donor:", type=["smi","mol","sdf"], key="donor_file")
            if file:
                donor = read_molecule_file(file)
        else:
            html = """
            <script src="https://unpkg.com/jsme-editor"></script>
            <div id="jsme_d" style="width:400px; height:350px;"></div>
            <script>
                var jsme_d = new JSApplet.JSME("jsme_d", "400px", "350px", {"options":"query"});
                function sendD(){window.parent.postMessage({don_jsme:jsme_d.smiles()}, "*");}
                setInterval(sendD, 800);
            </script>
            """
            components.html(html, height=380)
            js = st._get_script_run_ctx().session_state.get("don_jsme", "")
            if js:
                donor = js

    # -------- Acceptor -------
    with col2:
        st.subheader("Acceptor Molecule")
        method = st.radio("Input Method:", 
                          ["SMILES", "Draw (JSME)", "Upload"], key="acc_method")

        acceptor = ""
        if method == "SMILES":
            acceptor = st.text_input("Acceptor SMILES:", key="acc_smi")
        elif method == "Upload":
            file = st.file_uploader("Upload Acceptor:", type=["smi","mol","sdf"], key="acc_file")
            if file:
                acceptor = read_molecule_file(file)
        else:
            html = """
            <script src="https://unpkg.com/jsme-editor"></script>
            <div id="jsme_a" style="width:400px; height:350px;"></div>
            <script>
                var jsme_a = new JSApplet.JSME("jsme_a", "400px", "350px", {"options":"query"});
                function sendA(){window.parent.postMessage({acc_jsme:jsme_a.smiles()}, "*");}
                setInterval(sendA, 800);
            </script>
            """
            components.html(html, height=380)
            js = st._get_script_run_ctx().session_state.get("acc_jsme", "")
            if js:
                acceptor = js

    # ---------------- FRET logic -------------------
    if donor or acceptor:
        st.subheader("2D Structures")

        if donor:
            st.write("**Donor:**")
            show_mol(donor)

            # Predict donor emission + absorption
            ds1 = smiles_to_descriptors_df(donor)
            ds2 = smiles_to_descriptors_df("O")  # water
            if ds1 is not None and ds2 is not None:
                X = pd.concat([ds1, ds2], axis=1)
                donor_em = predict(model_em, X)
                donor_abs = predict(model_abs, X)
                st.info(f"Donor Emission: {donor_em:.1f} nm")
                st.info(f"Donor Absorption: {donor_abs:.1f} nm")

        if acceptor:
            st.write("**Acceptor:**")
            show_mol(acceptor)

            ds1 = smiles_to_descriptors_df(acceptor)
            ds2 = smiles_to_descriptors_df("O")
            if ds1 is not None and ds2 is not None:
                X = pd.concat([ds1, ds2], axis=1)
                acceptor_abs = predict(model_abs, X)
                st.info(f"Acceptor Absorption: {acceptor_abs:.1f} nm")

# -------------------------------------------------------
# Footer
# -------------------------------------------------------
st.write("---")
st.caption("FluroML © P. Deshmukh")
