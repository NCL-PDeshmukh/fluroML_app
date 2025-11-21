import os
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc

# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="FluroML - Molecular Fluorescence Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Global neon / dark scientific theme ----------
st.markdown(
    """
    <style>
    /* App background */
    .stApp {
        background: radial-gradient(circle at top, #050816 0, #020617 40%, #000 100%);
        color: #e5e7eb;
        font-family: "Segoe UI", system-ui, sans-serif;
    }
    /* Tweak default text */
    h1, h2, h3, h4 {
        color: #e5e7eb !important;
    }
    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("FluroML: Molecular Fluorescence Predictor")

# ======================================================
# KETCHER INTEGRATION
# ======================================================
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("⚠️ streamlit-ketcher is not installed. Please install it to draw molecules.")
        return ""

# ======================================================
# NEON GRADIENT HEADER HELPER
# ======================================================
def neon_header(icon: str, title: str, gradient: str):
    """
    icon: emoji like 🔎, 📈, 🔗
    title: header text
    gradient: CSS linear-gradient string
    """
    st.markdown(
        f"""
        <div style="
            margin: 0.3rem 0 1rem 0;
            padding: 0.75rem 1.1rem;
            border-radius: 14px;
            background: {gradient};
            display: flex;
            align-items: center;
            gap: 10px;
            box-shadow: 0 0 18px rgba(56,189,248,0.55);
            border: 1px solid rgba(148,163,184,0.4);
        ">
            <div style="font-size: 1.7rem;">{icon}</div>
            <div style="font-size: 1.1rem; font-weight: 600; letter-spacing: 0.02em;">
                {title}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ======================================================
# RDKit.js VIEWER (NO SERVER-SIDE DRAWING)
# ======================================================
def rdkit_viewer(smiles: str, key: str, height: int = 300):
    """Render molecule using RDKit.js (client-side, cloud-safe)."""
    if not smiles:
        return
    safe = smiles.replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <div id="rdkit_{key}" style="background: #020617; border-radius: 10px; padding: 8px;"></div>
    <script src="https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"></script>
    <script>
      initRDKitModule().then(function(RDKit) {{
        try {{
          var mol = RDKit.get_mol("{safe}");
          if (!mol) {{
            document.getElementById("rdkit_{key}").innerHTML =
              "<div style='color:#f97373;'>Invalid SMILES</div>";
            return;
          }}
          var svg = mol.get_svg();
          document.getElementById("rdkit_{key}").innerHTML = svg;
          mol.delete();
        }} catch (e) {{
          document.getElementById("rdkit_{key}").innerHTML =
            "<div style='color:#f97373;'>Drawing error: " + e + "</div>";
        }}
      }}).catch(function(e){{
        document.getElementById("rdkit_{key}").innerHTML =
          "<div style='color:#f97373;'>RDKit.js failed to load.</div>";
      }});
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ======================================================
# MODEL LOADING
# ======================================================
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# ======================================================
# FEATURE / PREDICTION HELPERS
# ======================================================
def smiles_to_morgan(smiles: str):
    """Morgan fingerprint (1024) for classifier."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    try:
        return featurizer.featurize([mol])[0]
    except Exception as fe:
        st.error(f"Failed to featurize molecule: {fe}")
        return None

def smiles_to_descriptors(smiles: str):
    """MACCS keys DataFrame row, for regression models."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    try:
        features = featurizer.featurize([mol])
    except Exception as fe:
        st.error(f"Failed to compute descriptors: {fe}")
        return None
    return pd.DataFrame(features)

def predict(model, features):
    """Safe prediction wrapper."""
    if model is None or features is None:
        return None
    import numpy as np
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        pred = model.predict(X)
    except Exception as pe:
        st.error(f"Prediction failed: {pe}")
        return None
    return pred[0] if hasattr(pred, "__len__") and not isinstance(pred, str) else pred

# ======================================================
# MOLECULE FILE READER
# ======================================================
def read_molecule_file(uploaded_file):
    """Read .smi / .mol / .sdf and return first SMILES."""
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    text = data.decode("utf-8", errors="ignore")

    if filename.lower().endswith(".smi"):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            st.error("SMI file is empty or invalid.")
            return None
        return lines[0].split()[0]

    if filename.lower().endswith(".mol"):
        mol = Chem.MolFromMolBlock(text)
        return Chem.MolToSmiles(mol) if mol else None

    if filename.lower().endswith(".sdf"):
        entries = [entry for entry in text.split("$$$$") if entry.strip()]
        if not entries:
            st.error("No molecules found in SDF file.")
            return None
        mol = Chem.MolFromMolBlock(entries[0])
        return Chem.MolToSmiles(mol) if mol else None

    st.error("Unsupported file format.")
    return None

# ======================================================
# DATASET FOR FRET
# ======================================================
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ======================================================
# TABS
# ======================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ======================================================
# TAB 1 – Fluorescence Classification
# ======================================================
with tab1:
    neon_header(
        "🔎",
        "Fluorescence Classification",
        "linear-gradient(90deg, #22d3ee, #4f46e5)"
    )

    input_method = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule (Ketcher)", "Upload File"),
        key="fluoro_method"
    )

    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="fluorescence_smiles")
    elif input_method == "Upload File":
        file = st.file_uploader(
            "Upload a molecule file (.smi, .mol, .sdf):",
            type=["smi", "mol", "sdf"],
            key="fluoro_file"
        )
        if file:
            smiles = read_molecule_file(file) or ""
    else:
        smiles = st_ketcher("")
        if smiles:
            st.write(f"**SMILES from drawing:** `{smiles}`")

    if smiles:
        rdkit_viewer(smiles, key="clf_view", height=260)
        if model_fluorescence is None:
            st.error("Fluorescence classification model is not loaded.")
        else:
            feats = smiles_to_morgan(smiles)
            if feats is not None:
                with st.spinner("Predicting fluorescence..."):
                    pred = predict(model_fluorescence, feats)
                if pred is not None:
                    st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# ======================================================
# TAB 2 – Absorption Max Prediction
# ======================================================
with tab2:
    neon_header(
        "📈",
        "Absorption Max Prediction (λ_ex)",
        "linear-gradient(90deg, #38bdf8, #a855f7)"
    )

    input_method2 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule (Ketcher)", "Upload File"),
        key="abs_method"
    )

    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="absorption_smiles")
    elif input_method2 == "Upload File":
        file2 = st.file_uploader(
            "Upload a molecule file (.smi, .mol, .sdf):",
            type=["smi", "mol", "sdf"],
            key="abs_file"
        )
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        abs_smiles = st_ketcher("")
        if abs_smiles:
            st.write(f"**SMILES from drawing:** `{abs_smiles}`")

    solvent = st.text_input(
        "Solvent SMILES (e.g., 'O' for water):",
        value="O",
        key="absorption_solvent"
    )

    if abs_smiles and solvent:
        rdkit_viewer(abs_smiles, key="abs_view", height=260)
        if model_regression is None:
            st.error("Absorption prediction model is not loaded.")
        else:
            desc_smiles = smiles_to_descriptors(abs_smiles)
            desc_solvent = smiles_to_descriptors(solvent)
            if desc_smiles is not None and desc_solvent is not None:
                feats = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting absorption maximum..."):
                    pred = predict(model_regression, feats)
                if pred is not None:
                    st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# ======================================================
# TAB 3 – Emission Max Prediction
# ======================================================
with tab3:
    neon_header(
        "📈",
        "Emission Max Prediction (λ_em)",
        "linear-gradient(90deg, #f97316, #ec4899)"
    )

    input_method3 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule (Ketcher)", "Upload File"),
        key="em_method"
    )

    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emission_smiles")
    elif input_method3 == "Upload File":
        file3 = st.file_uploader(
            "Upload a molecule file (.smi, .mol, .sdf):",
            type=["smi", "mol", "sdf"],
            key="em_file"
        )
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        em_smiles = st_ketcher("")
        if em_smiles:
            st.write(f"**SMILES from drawing:** `{em_smiles}`")

    solvent_em = st.text_input(
        "Solvent SMILES (e.g., 'O' for water):",
        value="O",
        key="emission_solvent"
    )

    if em_smiles and solvent_em:
        rdkit_viewer(em_smiles, key="em_view", height=260)
        if model_emission is None:
            st.error("Emission prediction model is not loaded.")
        else:
            desc_smiles = smiles_to_descriptors(em_smiles)
            desc_solvent = smiles_to_descriptors(solvent_em)
            if desc_smiles is not None and desc_solvent is not None:
                feats = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting emission maximum..."):
                    pred = predict(model_emission, feats)
                if pred is not None:
                    st.success(f"Predicted Emission Max: {pred:.2f} nm")

# ======================================================
# TAB 4 – FRET Analysis
# ======================================================
with tab4:
    neon_header(
        "🔗",
        "FRET Pair Analysis (Donor → Acceptor)",
        "linear-gradient(90deg, #22c55e, #0ea5e9)"
    )

    input_method4 = st.radio(
        "Donor Input Method:",
        ("SMILES Input", "Draw Molecule (Ketcher)", "Upload File"),
        key="fret_method"
    )

    donor_smiles = ""
    if input_method4 == "SMILES Input":
        donor_smiles = st.text_input(
            "Enter Donor Molecule SMILES:", key="fret_donor_smiles"
        )
    elif input_method4 == "Upload File":
        file4 = st.file_uploader(
            "Upload a donor molecule file (.smi, .mol, .sdf):",
            type=["smi", "mol", "sdf"],
            key="fret_file"
        )
        if file4:
            donor_smiles = read_molecule_file(file4) or ""
    else:
        donor_smiles = st_ketcher("")
        if donor_smiles:
            st.write(f"**SMILES from drawing:** `{donor_smiles}`")

    if donor_smiles:
        rdkit_viewer(donor_smiles, key="fret_view", height=260)

        if model_emission is None:
            st.error("Emission prediction model not loaded. Cannot perform FRET analysis.")
        else:
            df = load_dataset()
            if df is None:
                st.error("Dataset could not be loaded for FRET analysis.")
            else:
                required_cols = {
                    "Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"
                }
                if not required_cols.issubset(df.columns):
                    st.error("Dataset is missing required columns for FRET analysis.")
                else:
                    donor_desc = smiles_to_descriptors(donor_smiles)
                    solvent_desc = smiles_to_descriptors("O")  # water as solvent
                    if donor_desc is None or solvent_desc is None:
                        st.error("Failed to featurize donor or solvent.")
                    else:
                        feats = pd.concat([donor_desc, solvent_desc], axis=1)
                        with st.spinner("Predicting donor emission and scanning acceptors..."):
                            donor_emission = predict(model_emission, feats)

                        if donor_emission is None:
                            st.error("Failed to predict donor emission.")
                        else:
                            df_candidates = df[
                                df["Fluorescent labeling"].astype(str).str.lower().isin(
                                    ["yes", "true", "1"]
                                )
                            ].copy()
                            df_candidates = df_candidates[df_candidates["AbsorptioMax (nm)"].notna()]
                            df_candidates = df_candidates[df_candidates["Smiles"] != donor_smiles]

                            if df_candidates.empty:
                                st.error("No suitable acceptor candidates found.")
                            else:
                                df_candidates["abs_diff"] = (
                                    df_candidates["AbsorptioMax (nm)"] - donor_emission
                                ).abs()

                                best_per_smiles = df_candidates.loc[
                                    df_candidates.groupby("Smiles")["abs_diff"].idxmin()
                                ]
                                best_per_smiles = best_per_smiles.sort_values("abs_diff").reset_index(drop=True)

                                top_match = best_per_smiles.iloc[0]
                                best_smiles = top_match["Smiles"]
                                best_abs = float(top_match["AbsorptioMax (nm)"])
                                best_em = (
                                    float(top_match["EmissionMax (nm)"])
                                    if pd.notna(top_match["EmissionMax (nm)"])
                                    else None
                                )

                                fret_eff = (best_abs / (donor_emission + best_abs)) * 100

                                c1, c2 = st.columns(2)
                                with c1:
                                    rdkit_viewer(donor_smiles, key="fret_donor_view", height=240)
                                    st.write(f"**Predicted Donor Emission Max:** {donor_emission:.2f} nm")
                                with c2:
                                    rdkit_viewer(best_smiles, key="fret_acceptor_view", height=240)
                                    st.write(f"**Top Acceptor Absorption Max:** {best_abs:.2f} nm")
                                    if best_em is not None:
                                        st.write(f"**Top Acceptor Emission Max:** {best_em:.2f} nm")
                                    st.write(f"**Estimated FRET Efficiency:** {fret_eff:.2f}%")

                                top5 = best_per_smiles.head(5)[
                                    ["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "abs_diff"]
                                ].copy()
                                top5.rename(columns={
                                    "Smiles": "Acceptor SMILES",
                                    "AbsorptioMax (nm)": "Absorption (nm)",
                                    "EmissionMax (nm)": "Emission (nm)",
                                    "abs_diff": "Δ (nm)"
                                }, inplace=True)
                                top5["Absorption (nm)"] = top5["Absorption (nm)"].map(lambda x: f"{x:.2f}")
                                top5["Emission (nm)"] = top5["Emission (nm)"].map(
                                    lambda x: f"{x:.2f}" if str(x) != "nan" else "N/A"
                                )
                                top5["Δ (nm)"] = top5["Δ (nm)"].map(lambda x: f"{x:.2f}")

                                st.markdown("**Top 5 FRET Partner Candidates:**")
                                st.table(top5.reset_index(drop=True))

# ======================================================
# FOOTER
# ======================================================
st.write("---")
st.caption("FluroML © PDeshmukh")
