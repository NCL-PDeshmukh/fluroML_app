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
# Tab 4 — FRET Pair Analysis (updated)
# ---------------------------
import matplotlib.pyplot as plt

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** molecule for FRET compatibility and dataset-based spectral overlap search.")

    colD, colA = st.columns(2)

    # Donor input
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="f_d_method")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="f_d_smi")
        elif donor_method == "Upload File":
            df_up = st.file_uploader("Upload Donor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_d_file")
            if df_up:
                donor_smiles = read_molecule_file(df_up) or ""
        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
            donor_smiles = st.text_input("Paste Donor SMILES from drawing tool:", key="f_d_draw")

    # Acceptor input
    with colA:
        st.subheader("Acceptor Molecule")
        acc_method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="f_a_method")
        acceptor_smiles = ""
        if acc_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="f_a_smi")
        elif acc_method == "Upload File":
            af_up = st.file_uploader("Upload Acceptor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_a_file")
            if af_up:
                acceptor_smiles = read_molecule_file(af_up) or ""
        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES from drawing tool:", key="f_a_draw")

    # Warn if nothing
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule to begin FRET analysis.")
    else:
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded. Cannot perform FRET analysis.")
        else:
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            # show query structure
            render_rdkitjs(query_smiles, key="fret_query", height=280)

            # compute descriptors (MACCS pair with water)
            feats = make_macss_pair(query_smiles, "O")
            if feats is None:
                st.error("Descriptor computation failed for query molecule.")
            else:
                with st.spinner("Predicting spectral property..."):
                    if is_donor:
                        query_em = predict_model(model_emission, feats)
                        if query_em is None:
                            st.error("Donor emission prediction failed.")
                        else:
                            st.write(f"**Predicted Donor Emission Max:** {query_em:.2f} nm")
                    else:
                        query_abs = predict_model(model_regression, feats)
                        if query_abs is None:
                            st.error("Acceptor absorption prediction failed.")
                        else:
                            st.write(f"**Predicted Acceptor Absorption Max:** {query_abs:.2f} nm")

                # Load dataset and search
                st.markdown("### 🔍 Searching for best matching FRET partners...")
                df = load_dataset()

                if df is None:
                    st.info("FRET dataset not found (All Properties with Finguprints_3.csv). Dataset-based partner search skipped.")
                else:
                    # Validate required columns (we expect exact column names based on your CSV)
                    required_cols = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                    if not required_cols.issubset(set(df.columns)):
                        st.warning("FRET dataset missing one or more required columns; partner search skipped.")
                    else:
                        # filter to fluorescent entries
                        df_fluoro = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                        df_fluoro = df_fluoro[df_fluoro["Smiles"] != query_smiles]

                        if df_fluoro.empty:
                            st.info("No fluorescent candidates found in dataset after filtering.")
                        else:
                            if is_donor:
                                df_fluoro["Δ (nm)"] = (df_fluoro["AbsorptioMax (nm)"] - query_em).abs()
                                top_candidates = df_fluoro.sort_values("Δ (nm)").head(5).reset_index(drop=True)
                                st.markdown("### 🧩 Top 5 FRET Acceptor Candidates")
                            else:
                                df_fluoro["Δ (nm)"] = (df_fluoro["EmissionMax (nm)"] - query_abs).abs()
                                top_candidates = df_fluoro.sort_values("Δ (nm)").head(5).reset_index(drop=True)
                                st.markdown("### 🧩 Top 5 FRET Donor Candidates")

                            # Build display table with consistent column names
                            display_df = top_candidates[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ (nm)"]].copy()
                            display_df.rename(columns={
                                "Smiles": "SMILES",
                                "AbsorptioMax (nm)": "Absorption (nm)",
                                "EmissionMax (nm)": "Emission (nm)"
                            }, inplace=True)
                            # Format numeric columns
                            for c in ["Absorption (nm)", "Emission (nm)", "Δ (nm)"]:
                                display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                            st.table(display_df.reset_index(drop=True))

                            # Visualization (spectral overlaps)
                            wavelength = np.linspace(300, 800, 1000)
                            if is_donor:
                                donor_em_val = query_em
                                donor_curve = np.exp(-0.5 * ((wavelength - donor_em_val) / 20) ** 2)
                                for idx, row in top_candidates.iterrows():
                                    acc_abs = float(row["AbsorptioMax (nm)"])
                                    acc_curve = np.exp(-0.5 * ((wavelength - acc_abs) / 25) ** 2)
                                    overlap_area = np.trapz(np.minimum(donor_curve, acc_curve), wavelength)
                                    overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    ax.plot(wavelength, donor_curve, label=f"Donor Emission ({donor_em_val:.1f} nm)")
                                    ax.plot(wavelength, acc_curve, label=f"Acceptor Absorption ({acc_abs:.1f} nm)")
                                    ax.fill_between(wavelength, np.minimum(donor_curve, acc_curve), color="violet", alpha=0.3)
                                    ax.set_xlabel("Wavelength (nm)")
                                    ax.set_ylabel("Intensity")
                                    ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_val - acc_abs):.1f} nm")
                                    ax.legend()
                                    st.pyplot(fig)
                            else:
                                acceptor_abs_val = query_abs
                                acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs_val) / 25) ** 2)
                                for idx, row in top_candidates.iterrows():
                                    donor_em_i = float(row["EmissionMax (nm)"])
                                    donor_curve_i = np.exp(-0.5 * ((wavelength - donor_em_i) / 20) ** 2)
                                    overlap_area = np.trapz(np.minimum(donor_curve_i, acceptor_curve), wavelength)
                                    overlap_pct = overlap_area / np.trapz(donor_curve_i, wavelength) * 100
                                    fig, ax = plt.subplots(figsize=(6, 3))
                                    ax.plot(wavelength, donor_curve_i, label=f"Donor Emission ({donor_em_i:.1f} nm)")
                                    ax.plot(wavelength, acceptor_curve, label=f"Acceptor Absorption ({acceptor_abs_val:.1f} nm)")
                                    ax.fill_between(wavelength, np.minimum(donor_curve_i, acceptor_curve), color="violet", alpha=0.3)
                                    ax.set_xlabel("Wavelength (nm)")
                                    ax.set_ylabel("Intensity")
                                    ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_i - acceptor_abs_val):.1f} nm")
                                    ax.legend()
                                    st.pyplot(fig)

# Footer
st.write("---")
st.caption("FluroML © Pooja Sanjay Deshmukh")
