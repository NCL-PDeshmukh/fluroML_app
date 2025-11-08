import streamlit as st
from rdkit import Chem
from rdkit.Chem import Draw
import joblib
import pandas as pd
import deepchem as dc

# Try to import streamlit_ketcher for drawing, provide fallback if unavailable
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("streamlit-ketcher is not installed. Please install it to draw molecules.")
        return ""

# Streamlit app configuration
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Models with caching
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

# Helper functions
def smiles_to_morgan(smiles: str):
    """Convert SMILES to Morgan fingerprint vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES string.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    try:
        features = featurizer.featurize([mol])[0]  # Returns a numpy array (1024,)
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
    # Wrap in DataFrame for easy concatenation
    return pd.DataFrame(features)

def draw_molecule(smiles: str):
    """Return an image of the molecule from SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None

def predict(model, features):
    """Make a prediction using a trained model and feature vector."""
    if model is None:
        return None
    import numpy as np
    # Convert features to numpy array and ensure 2D shape
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        pred = model.predict(X)
    except Exception as pe:
        st.error(f"Prediction failed: {pe}")
        return None
    # If the result is an array or list, return the first element
    return pred[0] if hasattr(pred, "__len__") and not isinstance(pred, str) else pred

def read_molecule_file(uploaded_file):
    """Read a molecule file (.mol, .sdf, .smi) and return a SMILES string."""
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    if filename.lower().endswith('.smi'):
        content = data.decode('utf-8', errors='ignore')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            st.error("SMI file is empty or invalid.")
            return None
        # Each non-empty line: first token is SMILES
        smiles = lines[0].split()[0]
        if len(lines) > 1:
            st.info("Multiple SMILES in file; using the first entry.")
        return smiles
    elif filename.lower().endswith('.mol'):
        mol_block = data.decode('utf-8', errors='ignore')
        mol = Chem.MolFromMolBlock(mol_block)
        if mol is None:
            st.error("Failed to parse MOL file.")
            return None
        return Chem.MolToSmiles(mol)
    elif filename.lower().endswith('.sdf'):
        content = data.decode('utf-8', errors='ignore')
        # SDF may contain multiple molecules separated by $$$$
        entries = [entry for entry in content.split('$$$$') if entry.strip()]
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

# Cache dataset loading for FRET analysis
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# Set up the app layout with tabs
st.title("FluroML: Molecular Fluorescence Predictor")
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# Tab 1: Fluorescence Classification
with tab1:
    st.markdown("## 🧪 Fluorescence Classification")
    input_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="fluoro_method")
    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="fluorescence_smiles")
    elif input_method == "Upload File":
        file = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fluoro_file")
        if file:
            smiles = read_molecule_file(file) or ""
    else:  # Draw Molecule
        smiles = st_ketcher("")  # open the drawing widget
        if smiles:
            st.write(f"**SMILES from drawing:** {smiles}")
    if smiles:
        if model_fluorescence:
            features = smiles_to_morgan(smiles)
            if features is not None:
                with st.spinner("Predicting fluorescence..."):
                    prediction = predict(model_fluorescence, features)
                st.image(draw_molecule(smiles), caption="Molecule Structure")
                if prediction is None:
                    st.error("Prediction could not be made.")
                else:
                    st.success("Fluorescent" if int(prediction) == 1 else "Non-Fluorescent")
        else:
            st.error("Fluorescence classification model is not loaded.")

# Tab 2: Absorption Max Prediction
with tab2:
    st.markdown("## 🌈 Absorption Max Prediction")
    input_method2 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="abs_method")
    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="absorption_smiles")
    elif input_method2 == "Upload File":
        file2 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="abs_file")
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:  # Draw Molecule
        abs_smiles = st_ketcher("")
        if abs_smiles:
            st.write(f"**SMILES from drawing:** {abs_smiles}")
    solvent = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="absorption_solvent")
    if abs_smiles and solvent:
        if model_regression:
            desc_smiles = smiles_to_descriptors(abs_smiles)
            desc_solvent = smiles_to_descriptors(solvent)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting absorption maximum..."):
                    prediction = predict(model_regression, features)
                st.image(draw_molecule(abs_smiles), caption="Molecule Structure")
                if prediction is not None:
                    st.write(f"**Predicted Absorption Max:** {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Absorption prediction model is not loaded.")

# Tab 3: Emission Max Prediction
with tab3:
    st.markdown("## 🔦 Emission Max Prediction")
    input_method3 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="em_method")
    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emission_smiles")
    elif input_method3 == "Upload File":
        file3 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="em_file")
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        em_smiles = st_ketcher("")
        if em_smiles:
            st.write(f"**SMILES from drawing:** {em_smiles}")
    solvent_em = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="emission_solvent")
    if em_smiles and solvent_em:
        if model_emission:
            desc_smiles = smiles_to_descriptors(em_smiles)
            desc_solvent = smiles_to_descriptors(solvent_em)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting emission maximum..."):
                    prediction = predict(model_emission, features)
                st.image(draw_molecule(em_smiles), caption="Molecule Structure")
                if prediction is not None:
                    st.write(f"**Predicted Emission Max:** {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Emission prediction model is not loaded.")

# ======================================
# 🔬 TAB 4: FRET Pair Analysis (Updated)
# ======================================
import numpy as np
import matplotlib.pyplot as plt

with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** molecule for FRET compatibility and dataset-based spectral overlap search.")

    colD, colA = st.columns(2)

    # ----- Donor Input -----
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="fret_donor_method")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="fret_donor_smiles")
        elif donor_method == "Upload File":
            donor_file = st.file_uploader("Upload Donor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_donor_file")
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""
        else:
            donor_smiles = st_ketcher("")
            if donor_smiles:
                st.write(f"**Donor SMILES:** {donor_smiles}")

    # ----- Acceptor Input -----
    with colA:
        st.subheader("Acceptor Molecule")
        acceptor_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="fret_acceptor_method")
        acceptor_smiles = ""
        if acceptor_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="fret_acceptor_smiles")
        elif acceptor_method == "Upload File":
            acceptor_file = st.file_uploader("Upload Acceptor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_acceptor_file")
            if acceptor_file:
                acceptor_smiles = read_molecule_file(acceptor_file) or ""
        else:
            acceptor_smiles = st_ketcher("")
            if acceptor_smiles:
                st.write(f"**Acceptor SMILES:** {acceptor_smiles}")

    # ----- Single Molecule FRET Match Mode -----
    if donor_smiles or acceptor_smiles:
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded. Cannot perform FRET analysis.")
        else:
            # Determine if Donor or Acceptor mode
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            desc = smiles_to_descriptors(query_smiles)
            solvent_desc = smiles_to_descriptors("O")  # assume water

            if desc is not None and solvent_desc is not None:
                with st.spinner("Predicting spectral property..."):
                    features = pd.concat([desc, solvent_desc], axis=1)
                    if is_donor:
                        query_em = predict(model_emission, features)
                        st.image(draw_molecule(query_smiles), caption="Donor Molecule")
                        st.write(f"**Predicted Donor Emission Max:** {query_em:.2f} nm")
                    else:
                        query_abs = predict(model_regression, features)
                        st.image(draw_molecule(query_smiles), caption="Acceptor Molecule")
                        st.write(f"**Predicted Acceptor Absorption Max:** {query_abs:.2f} nm")

                # ---- Dataset-based FRET Partner Search ----
                st.markdown("### 🔍 Searching for best matching FRET partners...")
                df = load_dataset()

                if df is not None and {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}.issubset(df.columns):
                    df_fluoro = df[df['Fluorescent labeling'].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                    df_fluoro = df_fluoro[df_fluoro['Smiles'] != query_smiles]

                    if is_donor:
                        df_fluoro['Δ (nm)'] = (df_fluoro['AbsorptioMax (nm)'] - query_em).abs()
                        top_candidates = df_fluoro.sort_values('Δ (nm)').head(5)
                        top_candidates.rename(columns={
                            'Smiles': 'Acceptor SMILES',
                            'AbsorptioMax (nm)': 'Absorption (nm)',
                            'EmissionMax (nm)': 'Emission (nm)'
                        }, inplace=True)
                        st.markdown("### 🧩 Top 5 FRET Acceptor Candidates")
                    else:
                        df_fluoro['Δ (nm)'] = (df_fluoro['EmissionMax (nm)'] - query_abs).abs()
                        top_candidates = df_fluoro.sort_values('Δ (nm)').head(5)
                        top_candidates.rename(columns={
                            'Smiles': 'Donor SMILES',
                            'AbsorptioMax (nm)': 'Absorption (nm)',
                            'EmissionMax (nm)': 'Emission (nm)'
                        }, inplace=True)
                        st.markdown("### 🧩 Top 5 FRET Donor Candidates")

                    st.table(top_candidates.reset_index(drop=True))

                    # Optional visualization
                    if is_donor:
                        donor_em = query_em
                        for idx, row in top_candidates.iterrows():
                            acceptor_abs = row["Absorption (nm)"]
                            wavelength = np.linspace(300, 800, 1000)
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em) / 20) ** 2)
                            acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs) / 25) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label="Donor Emission", lw=2)
                            ax.plot(wavelength, acceptor_curve, label=f"Acceptor Absorption ({row['Absorption (nm)']:.1f} nm)", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), color="violet", alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em - acceptor_abs):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        acceptor_abs = query_abs
                        for idx, row in top_candidates.iterrows():
                            donor_em = row["Emission (nm)"]
                            wavelength = np.linspace(300, 800, 1000)
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em) / 20) ** 2)
                            acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs) / 25) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label=f"Donor Emission ({row['Emission (nm)']:.1f} nm)", lw=2)
                            ax.plot(wavelength, acceptor_curve, label="Acceptor Absorption", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), color="violet", alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em - acceptor_abs):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                else:
                    st.info("Dataset not available or missing necessary columns for FRET partner search.")
    else:
        st.warning("Please provide at least a Donor or an Acceptor molecule to begin FRET analysis.")


# Footer
st.write("---")
st.caption("FluroML-©PDeshmukh")
