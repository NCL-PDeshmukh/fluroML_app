# FluroML_2026.py
# Cloud-safe Streamlit app with Kekule.js 2D viewer and enriched fingerprint+descriptor representations

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
    # escape quotes/newlines to embed safely in JS
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
              var mol = Kekule.IO.loadFormatData("{s_escaped}", "smi");
              var viewer = new Kekule.ChemWidget.Viewer(document.getElementById('molviewer'));
              viewer.setChemObj(mol);
              viewer.setPredefinedSetting('moleculeView');
              // 2D layout so it looks like typical chem figure
              var layout = new Kekule.StructureLayout.LayoutFactory.createLayout('ForceDirected2D');
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
# JSME editor (export via query param)
# -------------------------
def jsme_editor(key: str, width: int = 520, height: int = 360, initial_smiles: str = "") -> str:
    """
    Render a JSME editor (full toolbar). On clicking 'Export' the page reloads with query param jsme_<key>=<smiles>.
    Returns the SMILES string from the query param if present, else empty string.
    """
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
# Enriched feature functions (Morgan FP + MACCS)
# -------------------------
# Morgan (ECFP-like) + MACCS concatenation -> enriched vector (1190 dims)
morgan_featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
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

def enriched_representation(smiles: str, solvent_smiles: str = None):
    """
    Returns numpy vector:
      - If solvent_smiles is None: concat(morgan (1024,), maccs (166,)) -> (1190,)
      - If solvent_smiles provided: concat(morgan, maccs_mol, maccs_solvent) -> (1356,)
    """
    fp = compute_morgan(smiles)
    desc = compute_maccs(smiles)
    if fp is None or desc is None:
        return None
    if solvent_smiles:
        solvent_desc = compute_maccs(solvent_smiles)
        if solvent_desc is None:
            return None
        return np.concatenate([fp, desc, solvent_desc]).reshape(1, -1)
    return np.concatenate([fp, desc]).reshape(1, -1)

# -------------------------
# File reader helpers
# -------------------------
def read_molecule_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    try:
        s = data.decode('utf-8', errors='ignore')
    except Exception:
        s = None
    if name.endswith('.smi'):
        if not s:
            return None
        first = [ln for ln in s.splitlines() if ln.strip()]
        if not first:
            return None
        return first[0].split()[0]
    elif name.endswith('.mol'):
        try:
            mol = Chem.MolFromMolBlock(data.decode('utf-8', errors='ignore'))
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None
    elif name.endswith('.sdf'):
        try:
            blocks = data.decode('utf-8', errors='ignore').split('$$$$')
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

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_absorption = load_model("new_best_regressor_compatible.joblib")
model_emission = load_model("best_regressor_emission_compatible.joblib")

def predict_model(model, feature_vector):
    if model is None or feature_vector is None:
        return None
    try:
        return model.predict(feature_vector)[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None

# -------------------------
# Dataset for FRET
# -------------------------
@st.cache_data
def load_dataset():
    try:
        df = pd.read_csv("All Properties with Finguprints_3.csv")
        return df
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")
        return None

# -------------------------
# App UI
# -------------------------
st.title("FluroML: Molecular Fluorescence Predictor (2D + Enriched Features)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -------------------------
# TAB 1: Fluorescence Classification
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
        show_mol_kekule(smiles, width=400, height=300)
        # enriched representation (no solvent)
        features = enriched_representation(smiles)
        if features is not None:
            pred = predict_model(model_fluorescence, features)
            if pred is None:
                st.error("Prediction failed.")
            else:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")
                st.write("Enriched (FP+Descriptors) vector length:", features.shape[1])

# -------------------------
# TAB 2: Absorption Prediction
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
        # enriched vector includes solvent MACCS appended
        features = enriched_representation(abs_smiles, solvent_smiles=solvent)
        if features is not None:
            pred = predict_model(model_absorption, features)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {float(pred):.2f} nm")
                st.write("Enriched vector shape (with solvent):", features.shape)

# -------------------------
# TAB 3: Emission Prediction
# -------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"], key="tab3_method")

    em_smiles = ""
    if method == "SMILES Input":
        em_smiles = st.text_input("Molecule SMILES:", key="tab3_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab3_upload")
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
        features = enriched_representation(em_smiles, solvent_smiles=solvent_em)
        if features is not None:
            pred = predict_model(model_emission, features)
            if pred is not None:
                st.success(f"Predicted Emission Max: {float(pred):.2f} nm")
                st.write("Enriched vector shape (with solvent):", features.shape)

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

    # FRET search (dataset-based)
    if donor_smiles or acceptor_smiles:
        df = load_dataset()
        if df is None:
            st.warning("FRET dataset not available.")
        else:
            solvent = "O"
            if donor_smiles:
                # predict donor emission and absorption using enriched representation with solvent
                feats = enriched_representation(donor_smiles, solvent_smiles=solvent)
                if feats is not None:
                    donor_em = predict_model(model_emission, feats)
                    donor_abs = predict_model(model_absorption, feats)
                    st.info(f"Donor Emission (pred): {donor_em:.1f} nm")
                    st.info(f"Donor Absorption (pred): {donor_abs:.1f} nm")

                    # search acceptors by Δ = |Acceptor Absorption - Donor Emission|
                    dff = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                    dff['Δ'] = (dff['AbsorptioMax (nm)'] - donor_em).abs()
                    top = dff.sort_values('Δ').head(5)
                    st.subheader("Top 5 Acceptor Candidates")
                    st.table(top[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])

            if acceptor_smiles:
                feats = enriched_representation(acceptor_smiles, solvent_smiles=solvent)
                if feats is not None:
                    acc_abs = predict_model(model_absorption, feats)
                    st.info(f"Acceptor Absorption (pred): {acc_abs:.1f} nm")

                    # search donors by Δ = |Donor Emission - Acceptor Absorption|
                    dff = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                    dff['Δ'] = (dff['EmissionMax (nm)'] - acc_abs).abs()
                    topd = dff.sort_values('Δ').head(5)
                    st.subheader("Top 5 Donor Candidates")
                    st.table(topd[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ']])

# -------------------------
# Footer
# -------------------------
st.write("---")
st.caption("FluroML © P. Deshmukh")
