# FluroML_2026_final.py
# Final cloud-ready Streamlit app matched to your model feature shapes:
# - Fluorescence model: Morgan FP (1024)
# - Absorption model: Molecule MACCS (166) + Solvent MACCS (166) => 332
# - Emission model: Molecule MACCS + Solvent MACCS => 332

import streamlit as st
import pandas as pd
import numpy as np
from rdkit import Chem
import joblib
import deepchem as dc
import urllib.parse
import streamlit.components.v1 as components

# -------------------------
# Page config
# -------------------------
st.set_page_config(page_title="FluroML: Molecular Fluorescence Predictor", layout="wide")

# -------------------------
# Kekule-based 2D viewer (cloud-safe)
# -------------------------
def show_mol_kekule(smiles: str, width=400, height=300):
    """Render a 2D structure in the browser using Kekule.js (no Python drawing libs)."""
    if not smiles:
        return
    s_escaped = smiles.replace('"', '\\"').replace('\n', '')
    html = f"""
    <html>
      <head>
        <script src="https://cdn.jsdelivr.net/npm/kekule@0.9.6/dist/kekule.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/kekule@0.9.6/dist/themes/default/kekule.css" rel="stylesheet" />
      </head>
      <body>
        <div id="molviewer" style="width:{width}px; height:{height}px; border: 1px solid #ddd;"></div>
        <script>
          (function(){{
            try {{
              var chemObj = Kekule.IO.loadFormatData("{s_escaped}", "smi");
              var viewer = new Kekule.ChemWidget.Viewer(document.getElementById('molviewer'));
              viewer.setChemObj(chemObj);
              viewer.setPredefinedSetting('moleculeView');
            }} catch(e) {{
              document.getElementById('molviewer').innerText = "Failed to render structure.";
            }}
          }})();
        </script>
      </body>
    </html>
    """
    components.html(html, height=height + 20)

# -------------------------
# JSME editor helper (export via query param)
# -------------------------
def jsme_editor(key: str, width: int = 520, height: int = 360, initial_smiles: str = "") -> str:
    param = f"jsme_{key}"
    q = st.experimental_get_query_params()
    if param in q:
        return q[param][0]

    jsme_js_url = "https://jsme-editor.github.io/dist/jsme.nocache.js"
    init_enc = urllib.parse.quote(initial_smiles or "")
    html = f"""
    <div id="jsme_container_{key}" style="border:1px solid #ddd; width:{width}px; height:{height}px;"></div>
    <div style="margin-top:6px;">
      <button id="export_{key}" style="margin-right:8px;">Export SMILES</button>
      <button id="clear_{key}">Clear</button>
    </div>
    <script src="{jsme_js_url}"></script>
    <script>
      (function() {{
        function initJSME() {{
          try {{
            var jsme = new JSApplet.JSME("jsme_container_{key}", "{width-20}px", "{height-70}px");
            var init = "{init_enc}";
            if (init) {{
              try {{ jsme.setSmiles(decodeURIComponent(init)); }} catch(e){{}}
            }}
            document.getElementById("export_{key}").onclick = function() {{
              var s = jsme.smiles();
              if(!s) {{ alert("No structure to export"); return; }}
              var enc = encodeURIComponent(s);
              var url = new URL(window.location.href);
              url.searchParams.set("{param}", enc);
              window.location.href = url.toString();
            }};
            document.getElementById("clear_{key}").onclick = function() {{
              try {{ jsme.setSmiles(""); }} catch(e){{}}
            }};
          }} catch(e) {{
            setTimeout(initJSME, 200);
          }}
        }}
        initJSME();
      }})();
    </script>
    """
    components.html(html, height=height + 80)
    return ""

# -------------------------
# Featurizers (DeepChem)
# -------------------------
# Morgan FP (size 1024)
morgan_featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
# MACCS keys (166)
maccs_featurizer = dc.feat.MACCSKeysFingerprint()

def compute_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = morgan_featurizer.featurize([mol])[0]
        return np.array(arr, dtype=float)
    except Exception as e:
        st.error(f"Morgan featurization error: {e}")
        return None

def compute_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = maccs_featurizer.featurize([mol])[0]
        return np.array(arr, dtype=float)
    except Exception as e:
        st.error(f"MACCS featurization error: {e}")
        return None

# -------------------------
# Feature building for each model (match training)
# -------------------------
def features_for_fluorescence(smiles: str):
    """Return shape (1,1024) numpy for fluorescence model (Morgan only)."""
    fp = compute_morgan(smiles)
    if fp is None:
        return None
    return fp.reshape(1, -1)

def features_for_absorption_or_emission(smiles: str, solvent_smiles: str):
    """
    Return shape (1,332) numpy: concat(maccs_mol (166), maccs_solvent (166))
    Matches Absorption & Emission model training (choice 3).
    """
    mol_desc = compute_maccs(smiles)
    solv_desc = compute_maccs(solvent_smiles)
    if mol_desc is None or solv_desc is None:
        return None
    return np.concatenate([mol_desc, solv_desc]).reshape(1, -1)

# -------------------------
# File reader helper
# -------------------------
def read_molecule_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        text = None

    if name.endswith(".smi"):
        if not text:
            return None
        first = [ln for ln in text.splitlines() if ln.strip()]
        if not first:
            return None
        return first[0].split()[0]
    elif name.endswith(".mol"):
        try:
            mol = Chem.MolFromMolBlock(text)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None
    elif name.endswith(".sdf"):
        try:
            blocks = text.split("$$$$")
            mol = Chem.MolFromMolBlock(blocks[0])
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None
    else:
        return None

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model load error ({path}): {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")   # expects 1024
model_absorption = load_model("new_best_regressor_compatible.joblib")  # expects 332
model_emission = load_model("best_regressor_emission_compatible.joblib")  # expects 332

def predict_model(model, feature_vector):
    if model is None or feature_vector is None:
        return None
    try:
        return model.predict(feature_vector)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------------
# Load dataset for FRET
# -------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")
        return None

# -------------------------
# App UI
# -------------------------
st.title("FluroML: Molecular Fluorescence Predictor (Final)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -------------------------
# TAB 1: Fluorescence Classification (Morgan 1024)
# -------------------------
with tab1:
    st.header("🧪 Fluorescence Classification")
    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"], key="tab1_method")
    smiles = ""

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="tab1_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab1_upload")
        if f:
            smiles = read_molecule_file(f)
    else:
        js = jsme_editor("tab1_jsme", width=520, height=360)
        if js:
            smiles = urllib.parse.unquote(js)
            st.success("Structure exported from JSME.")

    if smiles:
        st.subheader("Structure Preview")
        show_mol_kekule = show_mol_kekule if 'show_mol_kekule' in globals() else show_mol_kekule  # safe reference
        show_mol_kekule(smiles, width=400, height=300)

        feats = features_for_fluorescence(smiles)
        if feats is not None:
            pred = predict_model(model_fluorescence, feats)
            if pred is None:
                st.error("Prediction failed")
            else:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")
                st.write("Feature vector shape (used):", feats.shape)

# -------------------------
# TAB 2: Absorption Max Prediction (MACCS mol + MACCS solvent => 332)
# -------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")
    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"], key="tab2_method")
    abs_smiles = ""

    if method == "SMILES Input":
        abs_smiles = st.text_input("Molecule SMILES:", key="tab2_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab2_upload")
        if f:
            abs_smiles = read_molecule_file(f)
    else:
        js = jsme_editor("tab2_jsme", width=520, height=360)
        if js:
            abs_smiles = urllib.parse.unquote(js)

    solvent = st.text_input("Solvent SMILES (e.g., O):", key="tab2_solvent")

    if abs_smiles:
        st.subheader("Structure Preview")
        show_mol_kekule(abs_smiles, width=400, height=300)

    if abs_smiles and solvent:
        feats = features_for_absorption_or_emission(abs_smiles, solvent)
        if feats is not None:
            pred = predict_model(model_absorption, feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {float(pred):.2f} nm")
                st.write("Feature vector shape (used):", feats.shape)

# -------------------------
# TAB 3: Emission Max Prediction (MACCS mol + MACCS solvent => 332)
# -------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"], key="tab3_method")
    em_smiles = ""

    if method == "SMILES Input":
        em_smiles = st.text_input("Molecule SMILES:", key="tab3_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload file (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab3_upload")
        if f:
            em_smiles = read_molecule_file(f)
    else:
        js = jsme_editor("tab3_jsme", width=520, height=360)
        if js:
            em_smiles = urllib.parse.unquote(js)

    solvent_em = st.text_input("Solvent SMILES (e.g., O):", key="tab3_solvent")

    if em_smiles:
        st.subheader("Structure Preview")
        show_mol_kekule(em_smiles, width=400, height=300)

    if em_smiles and solvent_em:
        feats = features_for_absorption_or_emission(em_smiles, solvent_em)
        if feats is not None:
            pred = predict_model(model_emission, feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {float(pred):.2f} nm")
                st.write("Feature vector shape (used):", feats.shape)

# -------------------------
# TAB 4: FRET Analysis
# -------------------------
with tab4:
    st.header("🔬 FRET Analysis")
    colD, colA = st.columns(2)

    donor_smiles = ""
    acceptor_smiles = ""

    with colD:
        st.subheader("Donor")
        dmethod = st.radio("Donor Input Method:", ["SMILES", "Draw (JSME)", "Upload"], key="tab4_donor_method")
        if dmethod == "SMILES":
            donor_smiles = st.text_input("Donor SMILES:", key="tab4_donor_smi")
        elif dmethod == "Upload":
            f = st.file_uploader("Upload donor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab4_donor_up")
            if f:
                donor_smiles = read_molecule_file(f)
        else:
            js = jsme_editor("tab4_donor_jsme", width=480, height=320)
            if js:
                donor_smiles = urllib.parse.unquote(js)

    with colA:
        st.subheader("Acceptor")
        amethod = st.radio("Acceptor Input Method:", ["SMILES", "Draw (JSME)", "Upload"], key="tab4_acceptor_method")
        if amethod == "SMILES":
            acceptor_smiles = st.text_input("Acceptor SMILES:", key="tab4_acceptor_smi")
        elif amethod == "Upload":
            f = st.file_uploader("Upload acceptor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab4_acceptor_up")
            if f:
                acceptor_smiles = read_molecule_file(f)
        else:
            js = jsme_editor("tab4_acceptor_jsme", width=480, height=320)
            if js:
                acceptor_smiles = urllib.parse.unquote(js)

    if donor_smiles:
        st.write("### Donor structure")
        show_mol_kekule(donor_smiles, width=380, height=280)
    if acceptor_smiles:
        st.write("### Acceptor structure")
        show_mol_kekule(acceptor_smiles, width=380, height=280)

    # FRET dataset-based search
    if donor_smiles or acceptor_smiles:
        df = load_dataset()
        if df is None:
            st.warning("FRET dataset not available.")
        else:
            solvent = "O"
            if donor_smiles:
                feats = features_for_absorption_or_emission(donor_smiles, solvent)
                if feats is not None:
                    donor_em = predict_model(model_emission, feats)
                    donor_abs = predict_model(model_absorption, feats)
                    st.info(f"Donor Emission (pred): {donor_em:.1f} nm")
                    st.info(f"Donor Absorption (pred): {donor_abs:.1f} nm")

                    dff = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                    dff['Δ'] = (dff['AbsorptioMax (nm)'] - donor_em).abs()
                    top = dff.sort_values('Δ').head(5)
                    st.subheader("Top 5 Acceptor Candidates")
                    st.table(top[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])

            if acceptor_smiles:
                feats = features_for_absorption_or_emission(acceptor_smiles, solvent)
                if feats is not None:
                    acc_abs = predict_model(model_absorption, feats)
                    st.info(f"Acceptor Absorption (pred): {acc_abs:.1f} nm")

                    dff = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                    dff['Δ'] = (dff['EmissionMax (nm)'] - acc_abs).abs()
                    topd = dff.sort_values('Δ').head(5)
                    st.subheader("Top 5 Donor Candidates")
                    st.table(topd[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])

# Footer
st.write("---")
st.caption("FluroML © P. Deshmukh")
