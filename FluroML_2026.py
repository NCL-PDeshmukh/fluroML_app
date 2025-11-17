# FluroML_2026.py
# Final RDKit.js Streamlit app with updated FRET tab (uses dataset columns exactly)

import warnings
warnings.filterwarnings("ignore")

# Silence DeepChem/TF/Torch/RDKit noise
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DEEPLOG_LEVEL"] = "0"
os.environ["DEEPLOG_DISABLE"] = "1"
os.environ["DC_SILENCE_LOGGING"] = "1"
os.environ["KMP_WARNINGS"] = "FALSE"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["RDKIT_SILENCE_DEPRECATION_WARNINGS"] = "1"

import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem, RDLogger

# Silence RDKit logs
RDLogger.DisableLog('rdApp.*')

# Streamlit page config
st.set_page_config(page_title="FluroML: Molecular Fluorescence Predictor", layout="wide")
st.title("FluroML: Molecular Fluorescence Predictor")

# ---------------------------
# Models (cached)
# ---------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")       # expects Morgan 1024
model_regression   = load_model("new_best_regressor_compatible.joblib")    # expects MACCS(mol)+MACCS(solvent)
model_emission     = load_model("best_regressor_emission_compatible.joblib") # expects MACCS pair

# ---------------------------
# Featurizers (DeepChem)
# ---------------------------
_morgan = dc.feat.CircularFingerprint(radius=3, size=1024)
_maccs  = dc.feat.MACCSKeysFingerprint()

def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = _morgan.featurize([mol])[0]
        return np.array(arr, dtype=float).reshape(1, -1)
    except Exception as e:
        st.error(f"Morgan featurization failed: {e}")
        return None

def smiles_to_maccs_arr(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = _maccs.featurize([mol])[0]
        return np.array(arr, dtype=float)
    except Exception as e:
        st.error(f"MACCS featurizer failed: {e}")
        return None

def smiles_to_descriptors(smiles: str):
    """Return a one-row pandas DataFrame of MACCS features (keeps API like old code)."""
    arr = smiles_to_maccs_arr(smiles)
    if arr is None:
        return None
    cols = [f"maccs_{i}" for i in range(len(arr))]
    return pd.DataFrame([arr], columns=cols)

def make_macss_pair(mol_smiles: str, solvent_smiles: str):
    a = smiles_to_maccs_arr(mol_smiles)
    b = smiles_to_maccs_arr(solvent_smiles)
    if a is None or b is None:
        return None
    return np.concatenate([a, b]).reshape(1, -1)

# ---------------------------
# Prediction helper
# ---------------------------
def predict_model(model, features):
    if model is None or features is None:
        return None
    try:
        X = np.asarray(features)
        # models may expect 2D; ensure shape
        if X.ndim == 1:
            X = X.reshape(1, -1)
        return model.predict(X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# ---------------------------
# File reader helper
# ---------------------------
def read_molecule_file(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()
    try:
        text = data.decode("utf-8", errors="ignore")
    except Exception:
        return None
    if name.endswith(".smi") or name.endswith(".smiles"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        return lines[0].split()[0]
    if name.endswith(".mol"):
        m = Chem.MolFromMolBlock(text)
        return Chem.MolToSmiles(m) if m else None
    if name.endswith(".sdf"):
        blocks = text.split("$$$$")
        if not blocks:
            return None
        m = Chem.MolFromMolBlock(blocks[0])
        return Chem.MolToSmiles(m) if m else None
    return None

# ---------------------------
# Dataset loader for FRET
# ---------------------------
@st.cache_data
def load_dataset(path="All Properties with Finguprints_3.csv"):
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return None

# ---------------------------
# RDKit.js renderer (WASM) — local then CDN fallback
# ---------------------------
CDN_RDKIT_JS = "https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"

def render_rdkitjs(smiles: str, key: str, height: int = 360):
    """Embed RDKit.js and draw molecule as SVG. Attempts /RDKit_minimal.js (repo root), falls back to CDN."""
    if not smiles:
        components.html("<div>No SMILES provided</div>", height=80)
        return
    s = smiles.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "")
    html = f"""
    <div id="rdkit_container_{key}">Loading structure...</div>
    <script>
    (function(){{
      function insertScript(src, onload, onerror) {{
        var s = document.createElement('script');
        s.src = src;
        s.onload = onload;
        s.onerror = onerror;
        document.head.appendChild(s);
      }}
      insertScript("/RDKit_minimal.js", function(){{ initAndDraw(); }}, function(){{ 
         insertScript("{CDN_RDKIT_JS}", function(){{ initAndDraw(); }}, function(){{ 
            document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Could not load RDKit.js locally or from CDN. Upload RDKit_minimal.js + rdkit.wasm to repo.</div>";
         }});
      }});
      function initAndDraw() {{
        try {{
          if (typeof initRDKitModule !== 'undefined') {{
            initRDKitModule().then(function(RDKit){{ draw(RDKit); }});
          }} else if (typeof RDKit !== 'undefined' && RDKit.get_mol) {{
            draw(RDKit);
          }} else {{
            setTimeout(function(){{
               if (typeof RDKit !== 'undefined' && RDKit.get_mol) draw(RDKit);
               else document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>RDKit loaded but initialization not found.</div>";
            }}, 200);
          }}
        }} catch (e) {{
          document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>RDKit init error: " + e + "</div>";
        }}
      }}
      function draw(RDKit) {{
        try {{
          var mol = RDKit.get_mol("{s}");
          if (!mol) {{
            document.getElementById("rdkit_container_{key}").innerHTML = "<div>Invalid SMILES</div>";
            return;
          }}
          var svg = mol.get_svg();
          document.getElementById("rdkit_container_{key}").innerHTML = svg;
          mol.delete();
        }} catch (err) {{
          document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Drawing error: " + err + "</div>";
        }}
      }}
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ---------------------------
# UI tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ---------------------------
# Tab 1 — Classification
# ---------------------------
with tab1:
    st.header("🧪 Fluorescence Classification")
    input_method = st.radio("Input Method:", ("SMILES Input", "Draw", "Upload File"), key="clf_method")
    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="clf_smi")
    elif input_method == "Upload File":
        f = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="clf_file")
        if f:
            smiles = read_molecule_file(f) or ""
    else:
        st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
        smiles = st.text_input("Paste SMILES from drawing tool:", key="clf_drawn")

    if smiles:
        render_rdkitjs(smiles, key="clf_view", height=280)
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            pred = predict_model(model_fluorescence, feats)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# ---------------------------
# Tab 2 — Absorption prediction
# ---------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")
    input_method2 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="abs_method")
    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="abs_smi")
    elif input_method2 == "Upload File":
        f2 = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="abs_file")
        if f2:
            abs_smiles = read_molecule_file(f2) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        abs_smiles = st.text_input("Paste SMILES from drawing tool:", key="abs_drawn")

    solvent = st.text_input("Solvent SMILES (default 'O'):", value="O", key="abs_solv")

    if abs_smiles:
        render_rdkitjs(abs_smiles, key="abs_view", height=280)
    if abs_smiles and solvent:
        feats = make_macss_pair(abs_smiles, solvent)
        if feats is not None:
            pred = predict_model(model_regression, feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# ---------------------------
# Tab 3 — Emission prediction
# ---------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    input_method3 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="emi_method")
    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emi_smi")
    elif input_method3 == "Upload File":
        f3 = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="emi_file")
        if f3:
            em_smiles = read_molecule_file(f3) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        em_smiles = st.text_input("Paste SMILES from drawing tool:", key="emi_drawn")

    solvent_em = st.text_input("Solvent SMILES (default 'O'):", value="O", key="emi_solv")

    if em_smiles:
        render_rdkitjs(em_smiles, key="emi_view", height=280)
    if em_smiles and solvent_em:
        feats = make_macss_pair(em_smiles, solvent_em)
        if feats is not None:
            pred = predict_model(model_emission, feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {pred:.2f} nm")

# ---------------------------
# TAB 4 — FRET Pair Analysis (NO PLOTS, FAST MODE)
# ---------------------------

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis (Fast Mode)")
    st.markdown("Provide **Donor** or **Acceptor** molecule to compute predicted spectral properties and find top 5 FRET partners.")

    colD, colA = st.columns(2)

    # -------------------------
    # DONOR INPUT
    # -------------------------
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio(
            "Input Method:", 
            ("SMILES Input", "Draw (external)", "Upload File"),
            key="f_d_method"
        )
        donor_smiles = ""

        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="f_d_smi")

        elif donor_method == "Upload File":
            donor_file = st.file_uploader("Upload Donor (.smi/.mol/.sdf):",
                                          type=["smi","mol","sdf"],
                                          key="f_d_file")
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""

        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES.")
            donor_smiles = st.text_input("Paste Donor SMILES:", key="f_d_draw")

    # -------------------------
    # ACCEPTOR INPUT
    # -------------------------
    with colA:
        st.subheader("Acceptor Molecule")
        acc_method = st.radio(
            "Input Method:", 
            ("SMILES Input", "Draw (external)", "Upload File"),
            key="f_a_method"
        )
        acceptor_smiles = ""

        if acc_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="f_a_smi")

        elif acc_method == "Upload File":
            acc_file = st.file_uploader("Upload Acceptor (.smi/.mol/.sdf):",
                                        type=["smi","mol","sdf"],
                                        key="f_a_file")
            if acc_file:
                acceptor_smiles = read_molecule_file(acc_file) or ""

        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES.")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES:", key="f_a_draw")

    # -------------------------
    # VALIDATION
    # -------------------------
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule.")
        st.stop()

    if model_emission is None or model_regression is None:
        st.error("Models not loaded — cannot perform FRET analysis.")
        st.stop()

    # -------------------------
    # DETERMINE MODE
    # -------------------------
    is_donor = bool(donor_smiles)
    query = donor_smiles if is_donor else acceptor_smiles

    st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

    # Show structure
    render_rdkitjs(query, key="fret_query", height=240)

    # -------------------------
    # COMPUTE PREDICTED PROPERTIES
    # -------------------------

    feats = make_macss_pair(query, "O")  # use water
    if feats is None:
        st.error("Descriptor computation failed.")
        st.stop()

    with st.spinner("Predicting spectral properties..."):

        # Always compute both
        pred_abs = predict_model(model_regression, feats)      # predicted absorption
        pred_em  = predict_model(model_emission, feats)        # predicted emission

    # Show predicted results
    if is_donor:
        st.success(f"**Predicted Donor Absorption:** {pred_abs:.2f} nm")
        st.success(f"**Predicted Donor Emission:** {pred_em:.2f} nm")
    else:
        st.success(f"**Predicted Acceptor Absorption:** {pred_abs:.2f} nm")
        st.success(f"**Predicted Acceptor Emission:** {pred_em:.2f} nm")

    st.write("---")

    # -------------------------
    # DATASET LOADING
    # -------------------------
    df = load_dataset()
    if df is None:
        st.error("Dataset not available.")
        st.stop()

    required = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
    if not required.issubset(df.columns):
        st.error("Dataset missing required columns.")
        st.stop()

    # Filter fluorescent + remove self
    df_f = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes","true","1"])].copy()
    df_f = df_f[df_f["Smiles"] != query]

    if df_f.empty:
        st.warning("No fluorescent partners found.")
        st.stop()

    # -------------------------
    # COMPUTE Δ BETWEEN QUERY AND DATASET
    # -------------------------
    if is_donor:
        df_f["Δ (nm)"] = (df_f["AbsorptioMax (nm)"] - pred_em).abs()
    else:
        df_f["Δ (nm)"] = (df_f["EmissionMax (nm)"] - pred_abs).abs()

    top5 = df_f.sort_values("Δ (nm)").head(5).reset_index(drop=True)

    st.markdown("### 🧩 Top 5 FRET Partner Candidates")

    # -------------------------
    # BUILD WIDE RESULTS TABLE
    # -------------------------
    wide = pd.DataFrame({
        "Dataset SMILES": top5["Smiles"],
        "Dataset Absorption (nm)": top5["AbsorptioMax (nm)"],
        "Dataset Emission (nm)": top5["EmissionMax (nm)"],
        "Δ (nm)": top5["Δ (nm)"],
        "Predicted Query Absorption (nm)": [pred_abs]*5,
        "Predicted Query Emission (nm)": [pred_em]*5,
    })

    st.dataframe(wide, use_container_width=True)

    st.success("✓ FRET partner identification complete.")

# Footer
st.write("---")
st.caption("FluroML © PDeshmukh")
