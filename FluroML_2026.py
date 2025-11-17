# FluroML_2026.py
# RDKit.js Streamlit app with JSME drawing and FRET analysis
import os, warnings
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["DC_SILENCE_LOGGING"] = "1"
os.environ["RDKIT_SILENCE_DEPRECATION_WARNINGS"] = "1"

import streamlit as st
import streamlit.components.v1 as components
import joblib
import pandas as pd
import numpy as np
import deepchem as dc
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title="FluroML: Fluorescence Predictor (RDKit.js + JSME)", layout="wide")
st.title("FluroML: Molecular Fluorescence Predictor (RDKit.js + JSME)")

# -----------------------------
# Cached model loader
# -----------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.warning(f"Could not load model '{path}': {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# -----------------------------
# DeepChem featurizers
# -----------------------------
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

# -----------------------------
# Prediction helper
# -----------------------------
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

# -----------------------------
# Read molecule files
# -----------------------------
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
        mol = Chem.MolFromMolBlock(text)
        return Chem.MolToSmiles(mol) if mol else None
    if name.endswith(".sdf"):
        blocks = text.split("$$$$")
        if not blocks:
            return None
        mol = Chem.MolFromMolBlock(blocks[0])
        return Chem.MolToSmiles(mol) if mol else None
    return None

# -----------------------------
# Dataset loader for FRET
# -----------------------------
@st.cache_data
def load_dataset(path="All Properties with Finguprints_3.csv"):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

# -----------------------------
# RDKit.js renderer (local first, then CDN)
# -----------------------------
CDN_RDKIT_JS = "https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"

def render_rdkitjs(smiles: str, key: str, height: int = 320):
    if not smiles:
        components.html("<div>No SMILES</div>", height=80)
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
            document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Could not load RDKit.js locally or from CDN. Upload RDKit_minimal.js + rdkit.wasm to repo root.</div>"
         }});
      }});
      function initAndDraw() {{
        try {{
          if (typeof initRDKitModule !== 'undefined') {{
            initRDKitModule().then(function(RDKit){{ draw(RDKit); }});
          }} else if (typeof RDKit !== 'undefined' && RDKit.get_mol) {{
            draw(RDKit);
          }} else {{
            setTimeout(function(){{ if (typeof RDKit !== 'undefined' && RDKit.get_mol) draw(RDKit); else document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>RDKit loaded but initialization not found.</div>"; }}, 200);
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

# -----------------------------
# JSME embed (iframe). Users export SMILES from the editor and paste below.
# -----------------------------
JSME_IFRAME_SRC = "https://peter-ertl.com/jsme/jsme.html"

def jsme_embed(height=420):
    html = f"""
    <div style="font-size:14px; padding:6px; border-radius:6px; background:#eef6ff; margin-bottom:8px;">
      <strong>JSME editor:</strong> draw your molecule, click "Export SMILES", copy and paste the SMILES into the input field below.
    </div>
    <iframe src="{JSME_IFRAME_SRC}" width="100%" height="{height}" style="border:1px solid #ddd; border-radius:6px;"></iframe>
    """
    components.html(html, height=height+80, scrolling=True)

# -----------------------------
# Tabs
# -----------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# -----------------------------
# Tab 1: Classification
# -----------------------------
with tab1:
    st.header("🧪 Fluorescence Classification")
    method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="tab1_method")
    smiles = ""
    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="tab1_smi")
    elif method == "Draw Molecule (JSME)":
        jsme_embed(height=360)
        smiles = st.text_input("SMILES from JSME (paste here):", key="tab1_jsme")
    else:
        f = st.file_uploader("Upload (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab1_file")
        if f:
            smiles = read_molecule_file(f) or ""

    if smiles:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            st.error("Invalid SMILES")
        else:
            render_rdkitjs(smiles, key="tab1_rend", height=260)
            feats = smiles_to_morgan(smiles)
            if feats is not None:
                p = predict_model(model_fluorescence, feats)
                if p is not None:
                    st.success("Fluorescent" if int(p) == 1 else "Non-Fluorescent")

# -----------------------------
# Tab 2: Absorption
# -----------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")
    method2 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="tab2_method")
    abs_smiles = ""
    if method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter SMILES:", key="tab2_smi")
    elif method2 == "Draw Molecule (JSME)":
        jsme_embed(height=360)
        abs_smiles = st.text_input("SMILES from JSME (paste):", key="tab2_jsme")
    else:
        f2 = st.file_uploader("Upload (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab2_file")
        if f2:
            abs_smiles = read_molecule_file(f2) or ""

    solvent = st.text_input("Solvent SMILES (default 'O'):", value="O", key="tab2_solv")

    if abs_smiles:
        mol = Chem.MolFromSmiles(abs_smiles)
        if mol is None:
            st.error("Invalid SMILES")
        else:
            render_rdkitjs(abs_smiles, key="tab2_rend", height=260)
            feats = make_macss_pair(abs_smiles, solvent)
            if feats is not None:
                p = predict_model(model_regression, feats)
                if p is not None:
                    st.success(f"Predicted Absorption Max: {p:.2f} nm")

# -----------------------------
# Tab 3: Emission
# -----------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    method3 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="tab3_method")
    em_smiles = ""
    if method3 == "SMILES Input":
        em_smiles = st.text_input("Enter SMILES:", key="tab3_smi")
    elif method3 == "Draw Molecule (JSME)":
        jsme_embed(height=360)
        em_smiles = st.text_input("SMILES from JSME (paste):", key="tab3_jsme")
    else:
        f3 = st.file_uploader("Upload (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="tab3_file")
        if f3:
            em_smiles = read_molecule_file(f3) or ""

    solvent_em = st.text_input("Solvent SMILES (default 'O'):", value="O", key="tab3_solv")

    if em_smiles:
        mol = Chem.MolFromSmiles(em_smiles)
        if mol is None:
            st.error("Invalid SMILES")
        else:
            render_rdkitjs(em_smiles, key="tab3_rend", height=260)
            feats = make_macss_pair(em_smiles, solvent_em)
            if feats is not None:
                p = predict_model(model_emission, feats)
                if p is not None:
                    st.success(f"Predicted Emission Max: {p:.2f} nm")

# -----------------------------
# Tab 4: FRET analysis
# -----------------------------
import matplotlib.pyplot as plt

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide Donor or Acceptor (draw with JSME or paste SMILES), then search dataset for best partners.")

    colD, colA = st.columns(2)

    with colD:
        st.subheader("Donor")
        d_method = st.radio("Donor input:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="fret_d_method")
        donor_smiles = ""
        if d_method == "SMILES Input":
            donor_smiles = st.text_input("Donor SMILES:", key="fret_d_smi")
        elif d_method == "Draw Molecule (JSME)":
            jsme_embed(height=320)
            donor_smiles = st.text_input("Donor SMILES from JSME (paste):", key="fret_d_jsme")
        else:
            dfile = st.file_uploader("Upload donor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="fret_d_file")
            if dfile:
                donor_smiles = read_molecule_file(dfile) or ""

    with colA:
        st.subheader("Acceptor")
        a_method = st.radio("Acceptor input:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="fret_a_method")
        acceptor_smiles = ""
        if a_method == "SMILES Input":
            acceptor_smiles = st.text_input("Acceptor SMILES:", key="fret_a_smi")
        elif a_method == "Draw Molecule (JSME)":
            jsme_embed(height=320)
            acceptor_smiles = st.text_input("Acceptor SMILES from JSME (paste):", key="fret_a_jsme")
        else:
            afile = st.file_uploader("Upload acceptor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="fret_a_file")
            if afile:
                acceptor_smiles = read_molecule_file(afile) or ""

    if not (donor_smiles or acceptor_smiles):
        st.info("Please provide at least a Donor or Acceptor to start FRET analysis.")
    else:
        if model_emission is None or model_regression is None:
            st.error("Required models not loaded. FRET analysis cannot proceed.")
        else:
            is_donor = bool(donor_smiles)
            query = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")
            render_rdkitjs(query, key="fret_query", height=260)

            feats = make_macss_pair(query, "O")
            if feats is None:
                st.error("Failed to compute descriptors for query.")
            else:
                with st.spinner("Predicting spectral property..."):
                    if is_donor:
                        q_em = predict_model(model_emission, feats)
                        if q_em is None:
                            st.error("Donor emission prediction failed.")
                        else:
                            st.write(f"Predicted Donor Emission Max: {q_em:.2f} nm")
                    else:
                        q_abs = predict_model(model_regression, feats)
                        if q_abs is None:
                            st.error("Acceptor absorption prediction failed.")
                        else:
                            st.write(f"Predicted Acceptor Absorption Max: {q_abs:.2f} nm")

                st.markdown("### 🔍 Searching dataset for partners...")
                df = load_dataset()
                if df is None:
                    st.info("Dataset not found. Upload 'All Properties with Finguprints_3.csv' to enable partner search.")
                else:
                    required_cols = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                    if not required_cols.issubset(set(df.columns)):
                        st.warning("Dataset missing required columns; partner search skipped.")
                    else:
                        df_f = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                        df_f = df_f[df_f["Smiles"] != query]
                        if df_f.empty:
                            st.info("No fluorescent candidates after filtering.")
                        else:
                            if is_donor:
                                df_f["Δ (nm)"] = (df_f["AbsorptioMax (nm)"] - q_em).abs()
                                top = df_f.sort_values("Δ (nm)").head(5).reset_index(drop=True)
                                st.markdown("### Top 5 Acceptors")
                            else:
                                df_f["Δ (nm)"] = (df_f["EmissionMax (nm)"] - q_abs).abs()
                                top = df_f.sort_values("Δ (nm)").head(5).reset_index(drop=True)
                                st.markdown("### Top 5 Donors")

                            display = top[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ (nm)"]].copy()
                            display.rename(columns={"Smiles":"SMILES","AbsorptioMax (nm)":"Absorption (nm)","EmissionMax (nm)":"Emission (nm)"}, inplace=True)
                            for c in ["Absorption (nm)", "Emission (nm)", "Δ (nm)"]:
                                display[c] = display[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            st.table(display.reset_index(drop=True))

                            wavelength = np.linspace(300, 800, 1000)
                            if is_donor:
                                donor_curve = np.exp(-0.5 * ((wavelength - q_em) / 20) ** 2)
                                for _, row in top.iterrows():
                                    acc_abs = float(row["AbsorptioMax (nm)"])
                                    acc_curve = np.exp(-0.5 * ((wavelength - acc_abs) / 25) ** 2)
                                    overlap = np.trapz(np.minimum(donor_curve, acc_curve), wavelength)
                                    overlap_pct = overlap / np.trapz(donor_curve, wavelength) * 100
                                    fig, ax = plt.subplots(figsize=(6,3))
                                    ax.plot(wavelength, donor_curve, label=f"Donor Em ({q_em:.1f} nm)")
                                    ax.plot(wavelength, acc_curve, label=f"Acc Abs ({acc_abs:.1f} nm)")
                                    ax.fill_between(wavelength, np.minimum(donor_curve, acc_curve), color="violet", alpha=0.3)
                                    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
                                    ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(q_em-acc_abs):.1f} nm")
                                    ax.legend(); st.pyplot(fig)
                            else:
                                acc_curve = np.exp(-0.5 * ((wavelength - q_abs) / 25) ** 2)
                                for _, row in top.iterrows():
                                    donor_em_i = float(row["EmissionMax (nm)"])
                                    donor_curve_i = np.exp(-0.5 * ((wavelength - donor_em_i) / 20) ** 2)
                                    overlap = np.trapz(np.minimum(donor_curve_i, acc_curve), wavelength)
                                    overlap_pct = overlap / np.trapz(donor_curve_i, wavelength) * 100
                                    fig, ax = plt.subplots(figsize=(6,3))
                                    ax.plot(wavelength, donor_curve_i, label=f"Donor Em ({donor_em_i:.1f} nm)")
                                    ax.plot(wavelength, acc_curve, label=f"Acc Abs ({q_abs:.1f} nm)")
                                    ax.fill_between(wavelength, np.minimum(donor_curve_i, acc_curve), color="violet", alpha=0.3)
                                    ax.set_xlabel("Wavelength (nm)"); ax.set_ylabel("Intensity")
                                    ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_i-q_abs):.1f} nm")
                                    ax.legend(); st.pyplot(fig)

# Footer
st.write("---")
st.caption("FluroML © Pooja Sanjay Deshmukh")
