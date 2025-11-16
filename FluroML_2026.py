# FluroML_2026.py
# Final RDKit.js Streamlit app — server-side RDKit for features, client-side RDKit.js for 2D rendering

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
model_regression   = load_model("new_best_regressor_compatible.joblib")    # expects 332 (MACCS mol+solvent)
model_emission     = load_model("best_regressor_emission_compatible.joblib") # expects 332

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
    if name.endswith(".smi"):
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
        return pd.read_csv(path)
    except Exception:
        return None

# ---------------------------
# RDKit.js renderer (WASM) — tries local first, CDN fallback
# ---------------------------
# Recommended: upload RDKit_minimal.js and rdkit.wasm to repo root for reliability.
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
      // Try local file first
      insertScript("/RDKit_minimal.js", function(){{ initAndDraw(); }}, function(){{ 
         // local failed -> try CDN
         insertScript("{CDN_RDKIT_JS}", function(){{ initAndDraw(); }}, function(){{ 
            document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Could not load RDKit.js locally or from CDN. Upload RDKit_minimal.js + rdkit.wasm to repo.</div>"
         }});
      }});
      function initAndDraw() {{
        try {{
          // some builds expose initRDKitModule
          var promise;
          if (typeof initRDKitModule !== 'undefined') {{
            promise = initRDKitModule();
            promise.then(function(RDKit){{ draw(RDKit); }});
          }} else if (typeof RDKit !== 'undefined' && RDKit.get_mol) {{
            draw(RDKit);
          }} else {{
            // fallback attempt
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
# Layout with tabs
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
    method = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t1_method")
    smiles = ""
    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES:", key="t1_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="t1_file")
        if f:
            smiles = read_molecule_file(f) or ""
    else:
        st.info("Draw externally (JSME / Ketcher) and paste exported SMILES here.")
        smiles = st.text_input("Paste SMILES from your drawing tool:", key="t1_drawn")

    if smiles:
        render_rdkitjs(smiles, key="t1")
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            pred = predict_model(model_fluorescence, feats)
            if pred is None:
                st.error("Prediction failed.")
            else:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# ---------------------------
# Tab 2: Absorption Max Prediction
# ---------------------------
with tab2:
    st.header("🌈 Absorption Max Prediction")
    method2 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t2_method")
    abs_smiles = ""
    if method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="t2_smiles")
    elif method2 == "Upload File":
        f2 = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="t2_file")
        if f2:
            abs_smiles = read_molecule_file(f2) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        abs_smiles = st.text_input("Paste SMILES from drawing tool:", key="t2_drawn")

    solvent = st.text_input("Solvent SMILES (default O):", value="O", key="t2_solv")

    if abs_smiles:
        render_rdkitjs(abs_smiles, key="t2")
    if abs_smiles and solvent:
        feats = make_macss_pair(abs_smiles, solvent)
        if feats is not None:
            pred = predict_model(model_regression, feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# ---------------------------
# Tab 3: Emission Max Prediction
# ---------------------------
with tab3:
    st.header("🔦 Emission Max Prediction")
    method3 = st.radio("Input Method:", ("SMILES Input", "Draw (external)", "Upload File"), key="t3_method")
    em_smiles = ""
    if method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="t3_smiles")
    elif method3 == "Upload File":
        f3 = st.file_uploader("Upload molecule (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="t3_file")
        if f3:
            em_smiles = read_molecule_file(f3) or ""
    else:
        st.info("Draw externally and paste SMILES here.")
        em_smiles = st.text_input("Paste SMILES from drawing tool:", key="t3_drawn")

    solvent_em = st.text_input("Solvent SMILES (default O):", value="O", key="t3_solv")

    if em_smiles:
        render_rdkitjs(em_smiles, key="t3")
    if em_smiles and solvent_em:
        feats = make_macss_pair(em_smiles, solvent_em)
        if feats is not None:
            pred = predict_model(model_emission, feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {pred:.2f} nm")

# ---------------------------
# Tab 4: FRET Analysis (donor & acceptor)
# ---------------------------
with tab4:
    st.header("🔬 FRET Pair Analysis")
    st.markdown("Provide Donor and/or Acceptor SMILES. App predicts donor emission & donor absorption (if requested), acceptor absorption, and finds dataset matches.")

    col_d, col_a = st.columns(2)
    with col_d:
        donor_method = st.radio("Donor input:", ("SMILES Input", "Draw (external)", "Upload File"), key="f_d_method")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Donor SMILES:", key="f_d_smiles")
        elif donor_method == "Upload File":
            up = st.file_uploader("Upload donor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_d_file")
            if up:
                donor_smiles = read_molecule_file(up) or ""
        else:
            st.info("Draw externally & paste SMILES.")
            donor_smiles = st.text_input("Paste Donor SMILES:", key="f_d_drawn")

        # optional: compute donor absorption too
        donor_calc_abs = st.checkbox("Also compute predicted Donor Absorption (yes/no)", value=False, key="f_d_calc_abs")

    with col_a:
        acceptor_method = st.radio("Acceptor input:", ("SMILES Input", "Draw (external)", "Upload File"), key="f_a_method")
        acceptor_smiles = ""
        if acceptor_method == "SMILES Input":
            acceptor_smiles = st.text_input("Acceptor SMILES:", key="f_a_smiles")
        elif acceptor_method == "Upload File":
            up2 = st.file_uploader("Upload acceptor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_a_file")
            if up2:
                acceptor_smiles = read_molecule_file(up2) or ""
        else:
            st.info("Draw externally & paste SMILES.")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES:", key="f_a_drawn")

    # Show structures if present
    if donor_smiles:
        st.subheader("Donor Structure")
        render_rdkitjs(donor_smiles, key="f_d_view", height=300)
    if acceptor_smiles:
        st.subheader("Acceptor Structure")
        render_rdkitjs(acceptor_smiles, key="f_a_view", height=300)

    df = load_dataset()

    # Predictions
    donor_em = None
    donor_abs_pred = None
    acceptor_abs_pred = None

    if donor_smiles and model_emission is not None:
        feats = make_macss_pair(donor_smiles, "O")
        if feats is not None:
            donor_em = predict_model(model_emission, feats)
            if donor_em is not None:
                st.info(f"Predicted Donor Emission: {donor_em:.2f} nm")
    if donor_smiles and donor_calc_abs and model_regression is not None:
        feats_abs = make_macss_pair(donor_smiles, "O")
        if feats_abs is not None:
            donor_abs_pred = predict_model(model_regression, feats_abs)
            if donor_abs_pred is not None:
                st.info(f"Predicted Donor Absorption: {donor_abs_pred:.2f} nm")
    if acceptor_smiles and model_regression is not None:
        feats_acc = make_macss_pair(acceptor_smiles, "O")
        if feats_acc is not None:
            acceptor_abs_pred = predict_model(model_regression, feats_acc)
            if acceptor_abs_pred is not None:
                st.info(f"Predicted Acceptor Absorption: {acceptor_abs_pred:.2f} nm")

    # Search dataset for matches (if df available)
    if df is None:
        st.info("FRET dataset not found (All Properties with Finguprints_3.csv). Dataset-based partner search skipped.")
    else:
        # Ensure required columns
        required_cols = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
        if not required_cols.issubset(set(df.columns)):
            st.warning("FRET dataset missing required columns; partner search skipped.")
        else:
            # If donor provided => find acceptors
            if donor_smiles and donor_em is not None:
                df_cand = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                df_cand = df_cand[df_cand['Smiles'] != donor_smiles]
                df_cand = df_cand[df_cand['AbsorptioMax (nm)'].notna()]
                if df_cand.empty:
                    st.warning("No fluorescent acceptor candidates in dataset.")
                else:
                    df_cand['Δ (nm)'] = (df_cand['AbsorptioMax (nm)'] - donor_em).abs()
                    top = df_cand.sort_values('Δ (nm)').head(5).reset_index(drop=True)
                    st.subheader("Top 5 Acceptor Candidates")
                    st.table(top[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ (nm)']])

                    # plot overlap curves for top candidates
                    import matplotlib.pyplot as plt
                    wavelength = np.linspace(300, 800, 1000)
                    donor_curve = np.exp(-0.5*((wavelength - donor_em)/20)**2)
                    for i, row in top.iterrows():
                        acc_abs = row['AbsorptioMax (nm)']
                        acc_curve = np.exp(-0.5*((wavelength - acc_abs)/25)**2)
                        overlap_area = np.trapz(np.minimum(donor_curve, acc_curve), wavelength)
                        overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.plot(wavelength, donor_curve, label=f"Donor Emission ({donor_em:.1f} nm)")
                        ax.plot(wavelength, acc_curve, label=f"Acceptor Absorption ({acc_abs:.1f} nm)")
                        ax.fill_between(wavelength, np.minimum(donor_curve, acc_curve), alpha=0.3)
                        ax.set_xlabel("Wavelength (nm)")
                        ax.set_ylabel("Intensity (a.u.)")
                        ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em - acc_abs):.1f} nm")
                        ax.legend()
                        st.pyplot(fig)

            # If acceptor provided => find donors
            if acceptor_smiles and acceptor_abs_pred is not None:
                df_cand = df[df['Fluorescent labeling'].astype(str).str.lower().isin(['yes','true','1'])].copy()
                df_cand = df_cand[df_cand['Smiles'] != acceptor_smiles]
                df_cand = df_cand[df_cand['EmissionMax (nm)'].notna()]
                if df_cand.empty:
                    st.warning("No fluorescent donor candidates in dataset.")
                else:
                    df_cand['Δ (nm)'] = (df_cand['EmissionMax (nm)'] - acceptor_abs_pred).abs()
                    top = df_cand.sort_values('Δ (nm)').head(5).reset_index(drop=True)
                    st.subheader("Top 5 Donor Candidates")
                    st.table(top[['Smiles','AbsorptioMax (nm)','EmissionMax (nm)','Δ (nm)']])

                    wavelength = np.linspace(300, 800, 1000)
                    acc_curve = np.exp(-0.5*((wavelength - acceptor_abs_pred)/25)**2)
                    for i, row in top.iterrows():
                        donor_em_i = row['EmissionMax (nm)']
                        donor_curve_i = np.exp(-0.5*((wavelength - donor_em_i)/20)**2)
                        overlap_area = np.trapz(np.minimum(donor_curve_i, acc_curve), wavelength)
                        overlap_pct = overlap_area / np.trapz(donor_curve_i, wavelength) * 100
                        fig, ax = plt.subplots(figsize=(6,3))
                        ax.plot(wavelength, donor_curve_i, label=f"Donor Emission ({donor_em_i:.1f} nm)")
                        ax.plot(wavelength, acc_curve, label=f"Acceptor Absorption ({acceptor_abs_pred:.1f} nm)")
                        ax.fill_between(wavelength, np.minimum(donor_curve_i, acc_curve), alpha=0.3)
                        ax.set_xlabel("Wavelength (nm)")
                        ax.set_ylabel("Intensity (a.u.)")
                        ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_i - acceptor_abs_pred):.1f} nm")
                        ax.legend()
                        st.pyplot(fig)

# Footer
st.write("---")
st.caption("FluroML © Pooja Sanjay Deshmukh")
