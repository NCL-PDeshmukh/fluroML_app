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
# Tab 4 — FRET Pair Analysis (final updated with predictions)
# ---------------------------
import matplotlib.pyplot as plt
import numpy as np

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** molecule for FRET compatibility and dataset-based spectral overlap search.")

    colD, colA = st.columns(2)

    # Donor input
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio("Input Method (Donor):", ("SMILES Input", "Draw (external)", "Upload File"), key="f_d_method_v2")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="f_d_smi_v2")
        elif donor_method == "Upload File":
            df_up = st.file_uploader("Upload Donor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_d_file_v2")
            if df_up:
                donor_smiles = read_molecule_file(df_up) or ""
        else:
            st.info("Draw externally (e.g., JSME) and paste exported SMILES here.")
            donor_smiles = st.text_input("Paste Donor SMILES from drawing tool:", key="f_d_draw_v2")

    # Acceptor input
    with colA:
        st.subheader("Acceptor Molecule")
        acc_method = st.radio("Input Method (Acceptor):", ("SMILES Input", "Draw (external)", "Upload File"), key="f_a_method_v2")
        acceptor_smiles = ""
        if acc_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="f_a_smi_v2")
        elif acc_method == "Upload File":
            af_up = st.file_uploader("Upload Acceptor (.smi/.mol/.sdf):", type=["smi","mol","sdf"], key="f_a_file_v2")
            if af_up:
                acceptor_smiles = read_molecule_file(af_up) or ""
        else:
            st.info("Draw externally (e.g., JSME) and paste exported SMILES here.")
            acceptor_smiles = st.text_input("Paste Acceptor SMILES from drawing tool:", key="f_a_draw_v2")

    # Validate presence
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule to begin FRET analysis.")
    else:
        # Ensure models available
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded. Cannot perform FRET analysis.")
        else:
            # Mode selection & query
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            # Render query structure
            render_rdkitjs(query_smiles, key=f"fret_query_v2", height=280)

            # Compute predicted properties for the query molecule (both Abs & Em)
            query_feats = make_macss_pair(query_smiles, "O")
            if query_feats is None:
                st.error("Descriptor computation failed for query molecule.")
            else:
                with st.spinner("Predicting properties for query..."):
                    # Predicted absorption = regression model, predicted emission = emission model
                    query_pred_abs = predict_model(model_regression, query_feats)
                    query_pred_em  = predict_model(model_emission, query_feats)

                # Display query predictions
                colq1, colq2, colq3 = st.columns(3)
                with colq1:
                    if query_pred_abs is not None:
                        st.success(f"Predicted Absorption (query): {query_pred_abs:.2f} nm")
                    else:
                        st.info("Predicted Absorption (query): N/A")
                with colq2:
                    if query_pred_em is not None:
                        st.success(f"Predicted Emission (query): {query_pred_em:.2f} nm")
                    else:
                        st.info("Predicted Emission (query): N/A")
                with colq3:
                    st.caption("Note: predictions use solvent='O' (water) by default for descriptor pairing.")

                # Load dataset
                st.markdown("### 🔍 Searching dataset for best partners...")
                df = load_dataset()
                if df is None:
                    st.info("FRET dataset not found (All Properties with Finguprints_3.csv). Dataset-based partner search skipped.")
                else:
                    required_cols = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                    if not required_cols.issubset(set(df.columns)):
                        st.warning("FRET dataset missing required columns; partner search skipped.")
                    else:
                        # Filter fluorescent labeled entries and exclude query itself
                        df_fluoro = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes","true","1"])].copy()
                        df_fluoro = df_fluoro[df_fluoro["Smiles"] != query_smiles]
                        if df_fluoro.empty:
                            st.info("No fluorescent candidates found in dataset after filtering.")
                        else:
                            # For donor mode — find acceptors with dataset absorption closest to predicted donor emission
                            # For acceptor mode — find donors with dataset emission closest to predicted acceptor absorption
                            rows = []
                            for idx, row in df_fluoro.iterrows():
                                cand_smiles = row["Smiles"]
                                ds_abs = row.get("AbsorptioMax (nm)", np.nan)
                                ds_em  = row.get("EmissionMax (nm)", np.nan)

                                # Compute predicted properties for candidate
                                cand_feats = make_macss_pair(cand_smiles, "O")
                                if cand_feats is None:
                                    pred_abs = np.nan
                                    pred_em  = np.nan
                                else:
                                    pred_abs = predict_model(model_regression, cand_feats)
                                    pred_em  = predict_model(model_emission, cand_feats)

                                # Δ definition:
                                if is_donor:
                                    # Δ = |dataset absorption (candidate) - predicted donor emission|
                                    if pd.notna(ds_abs) and (query_pred_em is not None):
                                        delta = abs(float(ds_abs) - float(query_pred_em))
                                    else:
                                        delta = np.nan
                                else:
                                    # acceptor mode: Δ = |dataset emission (candidate donor) - predicted acceptor absorption|
                                    if pd.notna(ds_em) and (query_pred_abs is not None):
                                        delta = abs(float(ds_em) - float(query_pred_abs))
                                    else:
                                        delta = np.nan

                                rows.append({
                                    "SMILES": cand_smiles,
                                    "Dataset Absorption (nm)": ds_abs if pd.notna(ds_abs) else np.nan,
                                    "Dataset Emission (nm)": ds_em if pd.notna(ds_em) else np.nan,
                                    "Predicted Absorption (nm)": pred_abs if (pred_abs is not None) else np.nan,
                                    "Predicted Emission (nm)": pred_em if (pred_em is not None) else np.nan,
                                    "Δ (nm)": delta
                                })

                            # Build DataFrame and sort by Δ ascending (smallest spectral gap first)
                            df_candidates = pd.DataFrame(rows)
                            # drop rows where Δ is NaN for sorting, push them to bottom
                            df_candidates["Δ_sort"] = df_candidates["Δ (nm)"].apply(lambda x: x if not pd.isna(x) else 1e6)
                            df_candidates = df_candidates.sort_values("Δ_sort").drop(columns=["Δ_sort"]).reset_index(drop=True)

                            # Take top 5
                            top5 = df_candidates.head(5).copy()

                            # Format numeric columns for presentation
                            for col in ["Dataset Absorption (nm)", "Dataset Emission (nm)", "Predicted Absorption (nm)", "Predicted Emission (nm)", "Δ (nm)"]:
                                top5[col] = top5[col].apply(lambda x: f"{x:.2f}" if (pd.notna(x) and not np.isinf(x)) else "N/A")

                            st.markdown("### 🧩 Top 5 Candidates (dataset + predicted values)")
                            st.table(top5)

                            # For each candidate, show structure, dataset vs predicted values, and overlap plot
                            st.markdown("### 🔬 Candidate details and overlap plots")
                            wavelength = np.linspace(300, 800, 1000)

                            for i, cand in top5.reset_index(drop=True).iterrows():
                                csmi = cand["SMILES"]
                                ds_abs_val = cand["Dataset Absorption (nm)"]
                                ds_em_val  = cand["Dataset Emission (nm)"]
                                pred_abs_val = cand["Predicted Absorption (nm)"]
                                pred_em_val  = cand["Predicted Emission (nm)"]
                                delta_val = cand["Δ (nm)"]

                                st.markdown(f"#### Candidate {i+1}")
                                # show small two-column layout: structure + values
                                c1, c2 = st.columns([1, 2])
                                with c1:
                                    render_rdkitjs(csmi, key=f"fret_cand_{i}_view_v2", height=220)
                                with c2:
                                    st.write(f"**SMILES:** `{csmi}`")
                                    st.write(f"Dataset Absorption: **{ds_abs_val}** nm")
                                    st.write(f"Dataset Emission: **{ds_em_val}** nm")
                                    st.write(f"Predicted Absorption: **{pred_abs_val}** nm")
                                    st.write(f"Predicted Emission: **{pred_em_val}** nm")
                                    st.write(f"Δ (nm): **{delta_val}**")

                                # Overlap plot
                                # Convert strings like "N/A" back to floats where possible
                                try:
                                    if pred_em_val != "N/A":
                                        pred_em_float = float(pred_em_val)
                                    else:
                                        pred_em_float = None
                                except:
                                    pred_em_float = None
                                try:
                                    if pred_abs_val != "N/A":
                                        pred_abs_float = float(pred_abs_val)
                                    else:
                                        pred_abs_float = None
                                except:
                                    pred_abs_float = None
                                try:
                                    if ds_abs_val != "N/A":
                                        ds_abs_float = float(ds_abs_val)
                                    else:
                                        ds_abs_float = None
                                except:
                                    ds_abs_float = None
                                try:
                                    if ds_em_val != "N/A":
                                        ds_em_float = float(ds_em_val)
                                    else:
                                        ds_em_float = None
                                except:
                                    ds_em_float = None

                                fig, ax = plt.subplots(figsize=(7,3))
                                # Determine curves depending on mode
                                if is_donor:
                                    # donor emission = query_pred_em, acceptor absorption -> prefer predicted then dataset
                                    if query_pred_em is not None:
                                        donor_curve = np.exp(-0.5 * ((wavelength - float(query_pred_em)) / 20) ** 2)
                                        ax.plot(wavelength, donor_curve, label=f"Donor Em ({query_pred_em:.1f} nm)", lw=2)
                                    else:
                                        donor_curve = None
                                    # acceptor absorption: prefer dataset (ds_abs_float) then predicted (pred_abs_float)
                                    acc_x = pred_abs_float if pred_abs_float is not None else ds_abs_float
                                    if acc_x is not None:
                                        acc_curve = np.exp(-0.5 * ((wavelength - acc_x) / 25) ** 2)
                                        ax.plot(wavelength, acc_curve, label=f"Acceptor Abs ({acc_x:.1f} nm)", lw=2)
                                    else:
                                        acc_curve = None
                                    if donor_curve is not None and acc_curve is not None:
                                        overlap_area = np.trapz(np.minimum(donor_curve, acc_curve), wavelength)
                                        overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                                        ax.fill_between(wavelength, np.minimum(donor_curve, acc_curve), color="violet", alpha=0.3)
                                        ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={delta_val}")
                                    else:
                                        ax.set_title("Overlap: Insufficient numeric data to compute")
                                else:
                                    # acceptor mode: acceptor absorption = query_pred_abs, donors: prefer dataset emission then predicted emission
                                    if query_pred_abs is not None:
                                        acc_curve = np.exp(-0.5 * ((wavelength - float(query_pred_abs)) / 25) ** 2)
                                        ax.plot(wavelength, acc_curve, label=f"Acceptor Abs ({query_pred_abs:.1f} nm)", lw=2)
                                    else:
                                        acc_curve = None
                                    donor_x = pred_em_float if pred_em_float is not None else ds_em_float
                                    if donor_x is not None:
                                        donor_curve_i = np.exp(-0.5 * ((wavelength - donor_x) / 20) ** 2)
                                        ax.plot(wavelength, donor_curve_i, label=f"Donor Em ({donor_x:.1f} nm)", lw=2)
                                    else:
                                        donor_curve_i = None
                                    if acc_curve is not None and donor_curve_i is not None:
                                        overlap_area = np.trapz(np.minimum(donor_curve_i, acc_curve), wavelength)
                                        overlap_pct = overlap_area / np.trapz(donor_curve_i, wavelength) * 100
                                        ax.fill_between(wavelength, np.minimum(donor_curve_i, acc_curve), color="violet", alpha=0.3)
                                        ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={delta_val}")
                                    else:
                                        ax.set_title("Overlap: Insufficient numeric data to compute")

                                ax.set_xlabel("Wavelength (nm)")
                                ax.set_ylabel("Intensity")
                                ax.legend()
                                st.pyplot(fig)

# Footer
st.write("---")
st.caption("FluroML © PDeshmukh")
