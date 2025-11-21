import os
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc
from PIL import Image

# Try to import streamlit_ketcher for drawing
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("streamlit-ketcher is not installed. Please install it to draw molecules.")
        return ""

# ---------------------------
# Streamlit app configuration
# ---------------------------
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("FluroML: Molecular Fluorescence Predictor")

# ---------------------------
# Logo loader (optional)
# ---------------------------
def load_logo(path: str):
    if os.path.exists(path):
        try:
            return Image.open(path)
        except Exception:
            return None
    return None

logo_classification = load_logo("logo_classification.png")
logo_spectra        = load_logo("logo_spectra.png")
logo_fret           = load_logo("logo_fret.png")

# ---------------------------
# RDKit.js viewer (no server-side graphics)
# ---------------------------
def rdkit_viewer(smiles: str, key: str, height: int = 320):
    """
    Use RDKit.js in the browser to render an SVG.
    Works on Streamlit Cloud (no libXrender needed).
    """
    if not smiles:
        return
    safe = smiles.replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <div id="rdkit_{key}">Loading structure...</div>
    <script src="https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"></script>
    <script>
      initRDKitModule().then(function(RDKit) {{
        try {{
          var mol = RDKit.get_mol("{safe}");
          if (!mol) {{
            document.getElementById("rdkit_{key}").innerHTML = "<div>Invalid SMILES</div>";
            return;
          }}
          var svg = mol.get_svg();
          document.getElementById("rdkit_{key}").innerHTML = svg;
          mol.delete();
        }} catch (e) {{
          document.getElementById("rdkit_{key}").innerHTML = "<div style='color:red'>Drawing error: " + e + "</div>";
        }}
      }}).catch(function(e){{
        document.getElementById("rdkit_{key}").innerHTML = "<div style='color:red'>RDKit.js failed to load.</div>";
      }});
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ---------------------------
# Load Models with caching
# ---------------------------
@st.cache_resource
def load_model(path: str):
    try:
        model = joblib.load(path)
        return model
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# ---------------------------
# Helper functions
# ---------------------------
def smiles_to_morgan(smiles: str):
    """Convert SMILES to Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    try:
        features = featurizer.featurize([mol])[0]  # numpy array (1024,)
    except Exception as fe:
        st.error(f"Failed to featurize molecule: {fe}")
        return None
    return features

def smiles_to_descriptors(smiles: str):
    """Convert SMILES to MACCS key descriptors (returns a DataFrame row)."""
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
    """Make a prediction using a trained model and feature vector."""
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

def read_molecule_file(uploaded_file):
    """Read a molecule file (.mol, .sdf, .smi) and return a SMILES string."""
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    text = data.decode("utf-8", errors="ignore")

    if filename.lower().endswith('.smi'):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            st.error("SMI file is empty or invalid.")
            return None
        smiles = lines[0].split()[0]
        if len(lines) > 1:
            st.info("Multiple SMILES in file; using the first entry.")
        return smiles

    elif filename.lower().endswith('.mol'):
        mol = Chem.MolFromMolBlock(text)
        if mol is None:
            st.error("Failed to parse MOL file.")
            return None
        return Chem.MolToSmiles(mol)

    elif filename.lower().endswith('.sdf'):
        entries = [entry for entry in text.split('$$$$') if entry.strip()]
        if len(entries) == 0:
            st.error("No molecules found in SDF file.")
            return None
        mol = Chem.MolFromMolBlock(entries[0])
        if mol is None:
            st.error("Failed to parse SDF file.")
            return None
        if len(entries) > 1:
            st.info("Multiple molecules in SDF; using the first one.")
        return Chem.MolToSmiles(mol)

    else:
        st.error("Unsupported file format.")
        return None

# ---------------------------
# Cache dataset loading for FRET analysis
# ---------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

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
# Tab 1: Fluorescence Classification
# ===========================
with tab1:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_classification is not None:
            st.image(logo_classification, use_column_width=True)
        else:
            st.caption("logo_classification.png not found.")
    with col_title:
        st.markdown("## ΦF · Fluorescence Classification")

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
    else:  # Draw Molecule (Ketcher)
        smiles = st_ketcher("")
        if smiles:
            st.write(f"**SMILES from drawing:** `{smiles}`")

    if smiles:
        rdkit_viewer(smiles, key="clf_view")
        if model_fluorescence is None:
            st.error("Fluorescence classification model is not loaded.")
        else:
            features = smiles_to_morgan(smiles)
            if features is not None:
                with st.spinner("Predicting fluorescence..."):
                    prediction = predict(model_fluorescence, features)
                if prediction is None:
                    st.error("Prediction could not be made.")
                else:
                    st.success("Fluorescent" if int(prediction) == 1 else "Non-Fluorescent")

# ===========================
# Tab 2: Absorption Max Prediction
# ===========================
with tab2:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_spectra is not None:
            st.image(logo_spectra, use_column_width=True)
        else:
            st.caption("logo_spectra.png not found.")
    with col_title:
        st.markdown("## λ_ex · Absorption Max Prediction")

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
        "Enter Solvent SMILES (e.g., 'O' for water):",
        key="absorption_solvent",
        value="O"
    )

    if abs_smiles and solvent:
        rdkit_viewer(abs_smiles, key="abs_view")
        if model_regression is None:
            st.error("Absorption prediction model is not loaded.")
        else:
            desc_smiles = smiles_to_descriptors(abs_smiles)
            desc_solvent = smiles_to_descriptors(solvent)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting absorption maximum..."):
                    prediction = predict(model_regression, features)
                if prediction is not None:
                    st.success(f"Predicted Absorption Max: {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")

# ===========================
# Tab 3: Emission Max Prediction
# ===========================
with tab3:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_spectra is not None:
            st.image(logo_spectra, use_column_width=True)
        else:
            st.caption("logo_spectra.png not found.")
    with col_title:
        st.markdown("## λ_em · Emission Max Prediction")

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
        "Enter Solvent SMILES (e.g., 'O' for water):",
        key="emission_solvent",
        value="O"
    )

    if em_smiles and solvent_em:
        rdkit_viewer(em_smiles, key="em_view")
        if model_emission is None:
            st.error("Emission prediction model is not loaded.")
        else:
            desc_smiles = smiles_to_descriptors(em_smiles)
            desc_solvent = smiles_to_descriptors(solvent_em)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting emission maximum..."):
                    prediction = predict(model_emission, features)
                if prediction is not None:
                    st.success(f"Predicted Emission Max: {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")

# ===========================
# Tab 4: FRET Analysis
# ===========================
with tab4:
    col_logo, col_title = st.columns([1, 4])
    with col_logo:
        if logo_fret is not None:
            st.image(logo_fret, use_column_width=True)
        else:
            st.caption("logo_fret.png not found.")
    with col_title:
        st.markdown("## D→A · FRET Pair Analysis")

    input_method4 = st.radio(
        "Donor Input Method:",
        ("SMILES Input", "Draw Molecule (Ketcher)", "Upload File"),
        key="fret_method"
    )

    donor_smiles = ""
    if input_method4 == "SMILES Input":
        donor_smiles = st.text_input("Enter Donor Molecule SMILES:", key="fret_donor_smiles")
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
        rdkit_viewer(donor_smiles, key="fret_view")
        if model_emission is None:
            st.error("Emission prediction model not loaded. Cannot perform FRET analysis.")
        else:
            df = load_dataset()
            if df is None:
                st.error("Dataset could not be loaded for FRET analysis.")
            else:
                required_cols = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                if not required_cols.issubset(df.columns):
                    st.error("Dataset is missing required columns for FRET analysis.")
                else:
                    donor_desc = smiles_to_descriptors(donor_smiles)
                    solvent_desc = smiles_to_descriptors("O")  # water as solvent
                    if donor_desc is None or solvent_desc is None:
                        st.error("Failed to featurize donor or solvent.")
                    else:
                        features = pd.concat([donor_desc, solvent_desc], axis=1)
                        with st.spinner("Searching for optimal FRET acceptor..."):
                            donor_emission = predict(model_emission, features)

                        if donor_emission is None:
                            st.error("Failed to predict donor emission.")
                        else:
                            df_candidates = df[
                                df['Fluorescent labeling'].astype(str).str.lower().isin(["yes", "true", "1"])
                            ].copy()
                            df_candidates = df_candidates[df_candidates['AbsorptioMax (nm)'].notna()]
                            df_candidates = df_candidates[df_candidates['Smiles'] != donor_smiles]

                            if df_candidates.empty:
                                st.error("No suitable acceptor candidates found.")
                            else:
                                df_candidates['abs_diff'] = (
                                    df_candidates['AbsorptioMax (nm)'] - donor_emission
                                ).abs()

                                best_per_smiles = df_candidates.loc[
                                    df_candidates.groupby('Smiles')['abs_diff'].idxmin()
                                ]
                                best_per_smiles = best_per_smiles.sort_values('abs_diff').reset_index(drop=True)

                                top_match = best_per_smiles.iloc[0]
                                best_smiles = top_match['Smiles']
                                best_abs = float(top_match['AbsorptioMax (nm)'])
                                best_em = float(top_match['EmissionMax (nm)']) if pd.notna(
                                    top_match['EmissionMax (nm)']) else None

                                # Simplified FRET efficiency estimation
                                fret_eff = (best_abs / (donor_emission + best_abs)) * 100

                                col1, col2 = st.columns(2)
                                with col1:
                                    rdkit_viewer(donor_smiles, key="fret_donor_view", height=260)
                                    st.write(f"**Predicted Donor Emission Max:** {donor_emission:.2f} nm")
                                with col2:
                                    rdkit_viewer(best_smiles, key="fret_acceptor_view", height=260)
                                    st.write(f"**Acceptor Absorption Max:** {best_abs:.2f} nm")
                                    if best_em is not None:
                                        st.write(f"**Acceptor Emission Max:** {best_em:.2f} nm")
                                    st.write(f"**Estimated FRET Efficiency:** {fret_eff:.2f}%")

                                # Top 5 closest matches
                                top_n = 5
                                top_candidates = best_per_smiles.head(top_n)[
                                    ['Smiles', 'AbsorptioMax (nm)', 'EmissionMax (nm)', 'abs_diff']
                                ].copy()
                                top_candidates.rename(columns={
                                    'Smiles': 'Acceptor SMILES',
                                    'AbsorptioMax (nm)': 'Absorption (nm)',
                                    'EmissionMax (nm)': 'Emission (nm)',
                                    'abs_diff': 'Δ (nm)'
                                }, inplace=True)

                                top_candidates['Absorption (nm)'] = top_candidates['Absorption (nm)'].map(
                                    lambda x: f"{x:.2f}"
                                )
                                top_candidates['Emission (nm)'] = top_candidates['Emission (nm)'].map(
                                    lambda x: f"{x:.2f}" if str(x) != 'nan' else "N/A"
                                )
                                top_candidates['Δ (nm)'] = top_candidates['Δ (nm)'].map(
                                    lambda x: f"{x:.2f}"
                                )

                                st.markdown("**Top 5 Closest Matches:**")
                                st.table(top_candidates.reset_index(drop=True))

# Footer
st.write("---")
st.caption("FluroML-©PDeshmukh")
