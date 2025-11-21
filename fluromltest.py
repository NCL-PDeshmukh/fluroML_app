# FluroML_2026.py
# Streamlit app for fluorescence prediction + FRET
# Uses local JSME sketcher (no external network dependency) + logos.

import warnings
warnings.filterwarnings("ignore")

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
from PIL import Image

# Silence RDKit logs
RDLogger.DisableLog("rdApp.*")

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(
    page_title="FluroML: Molecular Fluorescence Predictor",
    layout="wide"
)
st.title("FluroML: Molecular Fluorescence Predictor")

# ---------------------------
# Logos
# ---------------------------
def load_logo(path: str):
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    return None

logo_classification = load_logo("logo_classification.png")   # search / classification icon
logo_spectra        = load_logo("logo_spectra.png")          # absorption/emission spectra
logo_fret           = load_logo("logo_fret.png")             # donor–acceptor FRET icon

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

model_fluorescence = load_model("best_classifier_compatible.joblib")          # Morgan 1024
model_regression   = load_model("new_best_regressor_compatible.joblib")       # MACCS(mol)+MACCS(solvent)
model_emission     = load_model("best_regressor_emission_compatible.joblib")  # MACCS pair

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
# RDKit.js renderer (WASM)
# ---------------------------
CDN_RDKIT_JS = "https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"

def render_rdkitjs(smiles: str, key: str, height: int = 360):
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
            document.getElementById("rdkit_container_{key}").innerHTML =
              "<div style='color:red'>Could not load RDKit.js. Add RDKit_minimal.js + rdkit.wasm or allow CDN.</div>";
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
               else document.getElementById("rdkit_container_{key}").innerHTML =
                 "<div style='color:red'>RDKit init not found.</div>";
            }}, 200);
          }}
        }} catch (e) {{
          document.getElementById("rdkit_container_{key}").innerHTML =
            "<div style='color:red'>RDKit init error: " + e + "</div>";
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
          document.getElementById("rdkit_container_{key}").innerHTML =
            "<div style='color:red'>Drawing error: " + err + "</div>";
        }}
      }}
    }})();
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ---------------------------
# JSME sketcher (local, offline)
# ---------------------------
def render_jsme_editor(key: str, height: int = 420):
    """
    Embed JSME molecular editor.

    IMPORTANT:
    - Place JSME files under .streamlit/static/jsme/
    - Streamlit serves that as /jsme/ in the browser
    """
    html = f"""
    <div id="jsme_container_{key}" style="width:100%; height:{height}px;"></div>

    <script type="text/javascript" src="/jsme/jsme.nocache.js"></script>

    <script type="text/javascript">
      // Called automatically by JSME after loading
      function jsmeOnLoad() {{
        var jsmeApplet = new JSApplet.JSME("jsme_container_{key}", "100%", "{height}px");
        window.jsme_{key} = jsmeApplet;
      }}
    </script>

    <p style="font-size:0.85rem; color:#444; margin-top:0.4rem;">
      Draw molecule above in <b>JSME</b> → Use JSME menu: <b>Export → SMILES</b> →
      copy the SMILES text and paste it into the input box below.
    </p>
    """
    components.html(html, height=height+100, scrolling=False)

# ---------------------------
# Tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ===========================
# Tab 1 — Classification
# ===========================
with tab1:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_classification is not None:
            st.image(logo_classification, use_column_width=True)
        else:
            st.caption("Logo 'logo_classification.png' not found.")
    with col_title:
        st.header("Fluorescence Classification")

    input_method = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw (JSME Editor)", "Upload File"),
        key="clf_method"
    )

    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="clf_smi")

    elif input_method == "Upload File":
        f = st.file_uploader(
            "Upload molecule (.smi/.mol/.sdf):",
            type=["smi", "mol", "sdf"],
            key="clf_file"
        )
        if f:
            smiles = read_molecule_file(f) or ""

    else:  # Draw (JSME Editor)
        render_jsme_editor(key="clf_jsme")
        smiles = st.text_input("Paste SMILES from JSME:", key="clf_drawn")

    if smiles:
        render_rdkitjs(smiles, key="clf_view", height=280)
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            pred = predict_model(model_fluorescence, feats)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# ===========================
# Tab 2 — Absorption
# ===========================
with tab2:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_spectra is not None:
            st.image(logo_spectra, use_column_width=True)
        else:
            st.caption("Logo 'logo_spectra.png' not found.")
    with col_title:
        st.header("Absorption Max Prediction")

    input_method2 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw (JSME Editor)", "Upload File"),
        key="abs_method"
    )

    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="abs_smi")

    elif input_method2 == "Upload File":
        f2 = st.file_uploader(
            "Upload molecule (.smi/.mol/.sdf):",
            type=["smi", "mol", "sdf"],
            key="abs_file"
        )
        if f2:
            abs_smiles = read_molecule_file(f2) or ""

    else:  # Draw (JSME Editor)
        render_jsme_editor(key="abs_jsme")
        abs_smiles = st.text_input("Paste SMILES from JSME:", key="abs_drawn")

    solvent = st.text_input("Solvent SMILES (default 'O'):", value="O", key="abs_solv")

    if abs_smiles:
        render_rdkitjs(abs_smiles, key="abs_view", height=280)

    if abs_smiles and solvent:
        feats = make_macss_pair(abs_smiles, solvent)
        if feats is not None:
            pred = predict_model(model_regression, feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# ===========================
# Tab 3 — Emission
# ===========================
with tab3:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_spectra is not None:
            st.image(logo_spectra, use_column_width=True)
        else:
            st.caption("Logo 'logo_spectra.png' not found.")
    with col_title:
        st.header("Emission Max Prediction")

    input_method3 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw (JSME Editor)", "Upload File"),
        key="emi_method"
    )

    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emi_smi")

    elif input_method3 == "Upload File":
        f3 = st.file_uploader(
            "Upload molecule (.smi/.mol/.sdf):",
            type=["smi", "mol", "sdf"],
            key="emi_file"
        )
        if f3:
            em_smiles = read_molecule_file(f3) or ""

    else:  # Draw (JSME Editor)
        render_jsme_editor(key="emi_jsme")
        em_smiles = st.text_input("Paste SMILES from JSME:", key="emi_drawn")

    solvent_em = st.text_input("Solvent SMILES (default 'O'):", value="O", key="emi_solv")

    if em_smiles:
        render_rdkitjs(em_smiles, key="emi_view", height=280)

    if em_smiles and solvent_em:
        feats = make_macss_pair(em_smiles, solvent_em)
        if feats is not None:
            pred = predict_model(model_emission, feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {pred:.2f} nm")

# ===========================
# Tab 4 — FRET
# ===========================
with tab4:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_fret is not None:
            st.image(logo_fret, use_column_width=True)
        else:
            st.caption("Logo 'logo_fret.png' not found.")
    with col_title:
        st.markdown("## FRET Pair Analysis (Fast Mode)")

    st.markdown(
        "Provide **Donor** or **Acceptor** molecule to compute predicted "
        "spectral properties and identify top 5 FRET partners."
    )

    colD, colA = st.columns(2)

    # Donor
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio(
            "Input Method:",
            ("SMILES Input", "Draw (JSME Editor)", "Upload File"),
            key="f_d_method"
        )
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="f_d_smi")
        elif donor_method == "Upload File":
            donor_file = st.file_uploader(
                "Upload Donor (.smi/.mol/.sdf):",
                type=["smi", "mol", "sdf"],
                key="f_d_file"
            )
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""
        else:  # Draw (JSME Editor)
            render_jsme_editor(key="f_d_jsme")
            donor_smiles = st.text_input("Paste Donor SMILES from JSME:", key="f_d_draw")

    # Acceptor
    with colA:
        st.subheader("Acceptor Molecule")
        acc_method = st.radio(
            "Input Method:",
            ("SMILES Input", "Draw (JSME Editor)", "Upload File"),
            key="f_a_method"
        )
        acceptor_smiles = ""
        if acc_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="f_a_smi")
        elif acc_method == "Upload File":
            acc_file = st.file_uploader(
                "Upload Acceptor (.smi/.mol/.sdf):",
                type=["smi", "mol", "sdf"],
                key="f_a_file"
            )
            if acc_file:
                acceptor_smiles = read_molecule_file(acc_file) or ""
        else:  # Draw (JSME Editor)
            render_jsme_editor(key="f_a_jsme")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES from JSME:", key="f_a_draw")

    # Validation
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule.")
        st.stop()

    if model_emission is None or model_regression is None:
        st.error("Models not loaded — cannot perform FRET analysis.")
        st.stop()

    is_donor = bool(donor_smiles)
    query = donor_smiles if is_donor else acceptor_smiles

    st.markdown(f"### Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

    # Show structure
    render_rdkitjs(query, key="fret_query", height=240)

    # Predicted properties
    feats = make_macss_pair(query, "O")  # water
    if feats is None:
        st.error("Descriptor computation failed.")
        st.stop()

    with st.spinner("Predicting spectral properties..."):
        pred_abs = predict_model(model_regression, feats)
        pred_em  = predict_model(model_emission, feats)

    if is_donor:
        st.success(f"Predicted Donor Absorption: {pred_abs:.2f} nm")
        st.success(f"Predicted Donor Emission: {pred_em:.2f} nm")
    else:
        st.success(f"Predicted Acceptor Absorption: {pred_abs:.2f} nm")
        st.success(f"Predicted Acceptor Emission: {pred_em:.2f} nm")

    st.write("---")

    # Dataset
    df = load_dataset()
    if df is None:
        st.error("Dataset not available.")
        st.stop()

    required = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
    if not required.issubset(df.columns):
        st.error("Dataset missing required columns.")
        st.stop()

    df_f = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
    df_f = df_f[df_f["Smiles"] != query]

    if df_f.empty:
        st.warning("No fluorescent partners found.")
        st.stop()

    if is_donor:
        df_f["Δ (nm)"] = (df_f["AbsorptioMax (nm)"] - pred_em).abs()
    else:
        df_f["Δ (nm)"] = (df_f["EmissionMax (nm)"] - pred_abs).abs()

    top5 = df_f.sort_values("Δ (nm)").head(5).reset_index(drop=True)

    st.markdown("### Top 5 FRET Partner Candidates")

    wide = pd.DataFrame({
        "Dataset SMILES": top5["Smiles"],
        "Dataset Absorption (nm)": top5["AbsorptioMax (nm)"],
        "Dataset Emission (nm)": top5["EmissionMax (nm)"],
        "Δ (nm)": top5["Δ (nm)"],
        "Predicted Query Absorption (nm)": [pred_abs] * 5,
        "Predicted Query Emission (nm)": [pred_em] * 5,
    })

    st.dataframe(wide, use_container_width=True)
    st.success("FRET partner identification complete.")

# ---------------------------
# Footer
# ---------------------------
st.write("---")
st.caption("FluroML © PDeshmukh")
