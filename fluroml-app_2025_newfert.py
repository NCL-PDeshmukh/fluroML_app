# FluroML_2026.py
# Final RDKit.js Streamlit app with updated FRET tab (Option A: RDKit.js SVG rendering)

import warnings
warnings.filterwarnings("ignore")

# Silence DeepChem / TF / RDKit warnings
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
from rdkit import Chem
from rdkit import RDLogger

# Silence RDKit logs
RDLogger.DisableLog('rdApp.*')

st.set_page_config(page_title="FluroML: Molecular Fluorescence Predictor (RDKit.js)", layout="wide")
st.title("FluroML: Molecular Fluorescence Predictor (RDKit.js)")

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

model_fluorescence = load_model("best_classifier_compatible.joblib")       # expects 1024 Morgan
model_regression   = load_model("new_best_regressor_compatible.joblib")    # expects MACCS(mol)+MACCS(solvent)
model_emission     = load_model("best_regressor_emission_compatible.joblib") # expects MACCS pair

# ---------------------------
# DeepChem featurizers
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

def smiles_to_maccs_df(smiles: str):
    """Return a one-row DataFrame of MACCS features (for compatibility with older code)."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        arr = _maccs.featurize([mol])[0]
        # return as DataFrame row (columns 0..len-1)
        cols = [f"maccs_{i}" for i in range(len(arr))]
        return pd.DataFrame([arr], columns=cols)
    except Exception as e:
        st.error(f"MACCS featurizer failed: {e}")
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
        return model.predict(np.asarray(features))[0]
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
# RDKit.js renderer (WASM) — tries local first, CDN fallback
# ---------------------------
# Recommended: upload RDKit_minimal.js and rdkit.wasm to repo root for reliability.
CDN_RDKIT_JS = "https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"

def render_rdkitjs(smiles: str, key: str, height: int = 360):
    """Embed RDKit.js in the page and draw the molecule as SVG."""
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
      // Try local file first
      insertScript("/RDKit_minimal.js", function(){{ initAndDraw(); }}, function(){{ 
         // local failed -> try CDN
         insertScript("{CDN_RDKIT_JS}", function(){{ initAndDraw(); }}, function(){{ 
            document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Could not load RDKit.js locally or from CDN. Upload RDKit_minimal.js + rdkit.wasm to repo.</div>"
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
# UI layout and tabs
# ---------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ---------------------------
# Tab 1: Fluorescence Classification
# ---------------------------
with tab1:
    st.header("🧪 Fluorescence Classification")
    input_method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t1_method")
    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="t1_smiles")
    elif input_method == "Upload File":
        file = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi","mol","sdf"], key="t1_file")
        if file:
            smiles = read_molecule_file(file) or ""
    else:
        st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
        smiles = st.text_input("Paste SMILES from drawing tool:", key="t1_drawn")

    if smiles:
        render_rdkitjs(smiles, key="render_t1", height=300)
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            prediction = predict_model(model_fluorescence, feats)
            if prediction is None:
                st.error("Prediction could not be made.")
            else:
                st.success("Fluorescent" if int(prediction) == 1 else "Non-Fluorescent")

# ---------------------------
# Tab 2: Absorption Max Prediction
# ---------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")
    input_method2 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t2_method")
    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="t2_smiles")
    elif input_method2 == "Upload File":
        file2 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi","mol","sdf"], key="t2_file")
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        abs_smiles = st.text_input("Paste SMILES from your drawing tool:", key="t2_drawn")

    solvent = st.text_input("Enter Solvent SMILES (default 'O'):", value="O", key="t2_solv")

    if abs_smiles:
        render_rdkitjs(abs_smiles, key="render_t2", height=300)
    if abs_smiles and solvent:
        feats = make_macss_pair(abs_smiles, solvent)
        if feats is not None:
            prediction = predict_model(model_regression, feats)
            if prediction is not None:
                st.success(f"**Predicted Absorption Max:** {prediction:.2f} nm")

# ---------------------------
# Tab 3: Emission Max Prediction
# ---------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    input_method3 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t3_method")
    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="t3_smiles")
    elif input_method3 == "Upload File":
        file3 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi","mol","sdf"], key="t3_file")
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        em_smiles = st.text_input("Paste SMILES from your drawing tool:", key="t3_drawn")

    solvent_em = st.text_input("Enter Solvent SMILES (default 'O'):", value="O", key="t3_solv")

    if em_smiles:
        render_rdkitjs(em_smiles, key="render_t3", height=300)
    if em_smiles and solvent_em:
        feats = make_macss_pair(em_smiles, solvent_em)
        if feats is not None:
            prediction = predict_model(model_emission, feats)
            if prediction is not None:
                st.success(f"**Predicted Emission Max:** {prediction:.2f} nm")

# ---------------------------
# ======================================
# 🔬 TAB 4: FRET Pair Analysis (Updated)
# ======================================
import matplotlib.pyplot as plt

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** molecule for FRET compatibility and dataset-based spectral overlap search.")

    colD, colA = st.columns(2)

    # ----- Donor Input -----
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="fret_donor_method")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="fret_donor_smiles")
        elif donor_method == "Upload File":
            donor_file = st.file_uploader("Upload Donor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_donor_file")
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""
        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
            donor_smiles = st.text_input("Paste Donor SMILES from drawing tool:", key="fret_donor_drawn")

    # ----- Acceptor Input -----
    with colA:
        st.subheader("Acceptor Molecule")
        acceptor_method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="fret_acceptor_method")
        acceptor_smiles = ""
        if acceptor_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="fret_acceptor_smiles")
        elif acceptor_method == "Upload File":
            acceptor_file = st.file_uploader("Upload Acceptor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_acceptor_file")
            if acceptor_file:
                acceptor_smiles = read_molecule_file(acceptor_file) or ""
        else:
            st.info("Draw externally (JSME/Ketcher) and paste exported SMILES here.")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES from drawing tool:", key="fret_acceptor_drawn")

    # If nothing provided warn
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule to begin FRET analysis.")
    else:
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded. Cannot perform FRET analysis.")
        else:
            # Determine mode
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            # Render query structure
            render_rdkitjs(query_smiles, key="fret_query", height=280)

            # Predict query property using MACCS pair (molecule + water)
            feats = make_macss_pair(query_smiles, "O")
            if feats is None:
                st.error("Failed to compute descriptors for query molecule.")
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

                # ---- Dataset-based FRET Partner Search ----
                st.markdown("### 🔍 Searching for best matching FRET partners...")
                df = load_dataset()

                if df is not None and {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}.issubset(df.columns):
                    df_fluoro = df[df['Fluorescent labeling'].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                    df_fluoro = df_fluoro[df_fluoro['Smiles'] != query_smiles]

                    if is_donor:
                        # compute absolute delta between donor emission and each acceptor absorption
                        df_fluoro['Δ (nm)'] = (df_fluoro['AbsorptioMax (nm)'] - query_em).abs()
                        top_candidates = df_fluoro.sort_values('Δ (nm)').head(5).reset_index(drop=True)
                        top_candidates.rename(columns={
                            'Smiles': 'Acceptor SMILES',
                            'AbsorptioMax (nm)': 'Absorption (nm)',
                            'EmissionMax (nm)': 'Emission (nm)'
                        }, inplace=True)
                        st.markdown("### 🧩 Top 5 FRET Acceptor Candidates")
                    else:
                        df_fluoro['Δ (nm)'] = (df_fluoro['EmissionMax (nm)'] - query_abs).abs()
                        top_candidates = df_fluoro.sort_values('Δ (nm)').head(5).reset_index(drop=True)
                        top_candidates.rename(columns={
                            'Smiles': 'Donor SMILES',
                            'AbsorptioMax (nm)': 'Absorption (nm)',
                            'EmissionMax (nm)': 'Emission (nm)'
                        }, inplace=True)
                        st.markdown("### 🧩 Top 5 FRET Donor Candidates")

                    # Display candidate table
                    display_df = top_candidates[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ (nm)']].copy()
                    display_df.columns = ['SMILES','Absorption (nm)','Emission (nm)','Δ (nm)']
                    # Format numeric columns if possible
                    for c in ['Absorption (nm)','Emission (nm)','Δ (nm)']:
                        display_df[c] = display_df[c].apply(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
                    st.table(display_df.reset_index(drop=True))

                    # Optional visualization: spectral overlap curves for each top candidate
                    if is_donor:
                        donor_em = query_em
                        wavelength = np.linspace(300, 800, 1000)
                        donor_curve = np.exp(-0.5 * ((wavelength - donor_em) / 20) ** 2)
                        for idx, row in top_candidates.iterrows():
                            acceptor_abs = float(row["Absorption (nm)"])
                            acceptor_em = row.get("Emission (nm)", None)
                            acceptor_em_val = float(acceptor_em) if pd.notna(acceptor_em) else None
                            wavelength = np.linspace(300, 800, 1000)
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em) / 20) ** 2)
                            acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs) / 25) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label="Donor Emission", lw=2)
                            ax.plot(wavelength, acceptor_curve, label=f"Acceptor Absorption ({acceptor_abs:.1f} nm)", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), color="violet", alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em - acceptor_abs):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        acceptor_abs = query_abs
                        wavelength = np.linspace(300, 800, 1000)
                        acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs) / 25) ** 2)
                        for idx, row in top_candidates.iterrows():
                            donor_em_val = float(row["Emission (nm)"])
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em_val) / 20) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label=f"Donor Emission ({donor_em_val:.1f} nm)", lw=2)
                            ax.plot(wavelength, acceptor_curve, label="Acceptor Absorption", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), color="violet", alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_val - acceptor_abs):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                else:
                    st.info("Dataset not available or missing necessary columns for FRET partner search.")

# Footer
st.write("---")
st.caption("FluroML © Pooja Sanjay Deshmukh")
