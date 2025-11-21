import os
import streamlit as st
import streamlit.components.v1 as components
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc

# Try to import streamlit_ketcher for drawing
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("streamlit-ketcher is not installed. Please install it to draw molecules.")
        return ""

# ----------------- Streamlit config -----------------
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Make tab labels a bit bigger & bold
st.markdown("""
<style>
div[data-baseweb="tab"] > button {
    font-size: 18px !important;
    font-weight: 700 !important;
}
</style>
""", unsafe_allow_html=True)

st.title("FluroML: Molecular Fluorescence Predictor")

# ----------------- RDKit.js structure viewer -----------------
def render_structure(smiles: str, key: str, height: int = 280):
    """Render molecule structure in-browser using RDKit.js (cloud-safe)."""
    if not smiles:
        return
    safe = smiles.replace("\\", "\\\\").replace('"', '\\"')
    html = f"""
    <div id="mol_view_{key}">Loading structure...</div>
    <script src="https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"></script>
    <script>
      initRDKitModule().then(function(RDKit) {{
        try {{
          const mol = RDKit.get_mol("{safe}");
          if (!mol) {{
            document.getElementById("mol_view_{key}").innerHTML =
              "<div style='color:#f97373;'>Invalid SMILES</div>";
            return;
          }}
          const svg = mol.get_svg();
          document.getElementById("mol_view_{key}").innerHTML = svg;
          mol.delete();
        }} catch (e) {{
          document.getElementById("mol_view_{key}").innerHTML =
            "<div style='color:#f97373;'>Drawing error: " + e + "</div>";
        }}
      }}).catch(function(e) {{
        document.getElementById("mol_view_{key}").innerHTML =
          "<div style='color:#f97373;'>RDKit.js failed to load.</div>";
      }});
    </script>
    """
    components.html(html, height=height, scrolling=False)

# ----------------- Model loading -----------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")        # Morgan
model_regression   = load_model("new_best_regressor_compatible.joblib")     # Absorption (MACCS pair)
model_emission     = load_model("best_regressor_emission_compatible.joblib")# Emission (MACCS pair)

# ----------------- Feature helpers -----------------
def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    return dc.feat.CircularFingerprint(radius=3, size=1024).featurize([mol])[0]

def smiles_to_maccs_arr(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    return dc.feat.MACCSKeysFingerprint().featurize([mol])[0]

def smiles_to_descriptors(smiles: str):
    """MACCS keys as DataFrame row (for older parts)."""
    arr = smiles_to_maccs_arr(smiles)
    if arr is None:
        return None
    return pd.DataFrame([arr])

def make_maccs_pair(mol_smiles: str, solvent_smiles: str):
    a = smiles_to_maccs_arr(mol_smiles)
    b = smiles_to_maccs_arr(solvent_smiles)
    if a is None or b is None:
        return None
    import numpy as np
    return np.concatenate([a, b]).reshape(1, -1)

def predict_array(model, features):
    """Model + numpy array 1xN."""
    if model is None or features is None:
        return None
    import numpy as np
    X = features
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        pred = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None
    return pred[0]

def predict_df(model, features_df: pd.DataFrame):
    """Model + DataFrame (for backward compatibility)."""
    if model is None or features_df is None:
        return None
    import numpy as np
    X = features_df.values
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        pred = model.predict(X)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None
    return pred[0]

# ----------------- File reader -----------------
def read_molecule_file(uploaded_file):
    text = uploaded_file.getvalue().decode("utf-8", errors="ignore")
    name = uploaded_file.name.lower()

    if name.endswith(".smi"):
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            st.error("SMI file is empty.")
            return None
        return lines[0].split()[0]

    if name.endswith(".mol"):
        mol = Chem.MolFromMolBlock(text)
        return Chem.MolToSmiles(mol) if mol else None

    if name.endswith(".sdf"):
        blocks = [blk for blk in text.split("$$$$") if blk.strip()]
        if not blocks:
            st.error("No molecules found in SDF.")
            return None
        mol = Chem.MolFromMolBlock(blocks[0])
        return Chem.MolToSmiles(mol) if mol else None

    st.error("Unsupported file format.")
    return None

# ----------------- Dataset for FRET -----------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ----------------- Tabs (icons in tab bar) -----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔎 Fluorescence Classification",
    "📈 Absorption Max Prediction",
    "📈 Emission Max Prediction",
    "🔗 FRET Analysis"
])

# =====================================================
# TAB 1 – Fluorescence Classification
# =====================================================
with tab1:
    method = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="clf_input_method"
    )
    smiles = ""

    if method == "SMILES Input":
        smiles = st.text_input("Enter SMILES", key="clf_smiles")
    elif method == "Upload File":
        file = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="clf_file"
        )
        if file:
            smiles = read_molecule_file(file) or ""
    else:
        smiles = st_ketcher("")
        if smiles:
            st.write(f"SMILES from drawing: `{smiles}`")

    if smiles:
        # Show structure for classification query
        render_structure(smiles, key="clf_mol", height=260)

    if smiles and model_fluorescence:
        feats = smiles_to_morgan(smiles)
        if feats is not None:
            pred = predict_array(model_fluorescence, feats)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")

# =====================================================
# TAB 2 – Absorption Max Prediction
# =====================================================
with tab2:
    method2 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="abs_input_method"
    )
    abs_smiles = ""
    solvent = st.text_input("Solvent SMILES", value="O", key="abs_solvent")

    if method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter SMILES", key="abs_smiles")
    elif method2 == "Upload File":
        file2 = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="abs_file"
        )
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        abs_smiles = st_ketcher("")
        if abs_smiles:
            st.write(f"SMILES from drawing: `{abs_smiles}`")

    if abs_smiles:
        # Show structure for absorption query
        render_structure(abs_smiles, key="abs_mol", height=260)

    if abs_smiles and solvent and model_regression:
        pair_feats = make_maccs_pair(abs_smiles, solvent)
        if pair_feats is not None:
            pred = predict_array(model_regression, pair_feats)
            if pred is not None:
                st.success(f"Predicted Absorption Max: {pred:.2f} nm")

# =====================================================
# TAB 3 – Emission Max Prediction
# =====================================================
with tab3:
    method3 = st.radio(
        "Input Method:",
        ("SMILES Input", "Draw Molecule", "Upload File"),
        key="emi_input_method"
    )
    em_smiles = ""
    solvent_em = st.text_input("Solvent SMILES", value="O", key="emi_solvent")

    if method3 == "SMILES Input":
        em_smiles = st.text_input("Enter SMILES", key="emi_smiles")
    elif method3 == "Upload File":
        file3 = st.file_uploader(
            "Upload molecule", type=["smi", "mol", "sdf"], key="emi_file"
        )
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        em_smiles = st_ketcher("")
        if em_smiles:
            st.write(f"SMILES from drawing: `{em_smiles}`")

    if em_smiles:
        # Show structure for emission query
        render_structure(em_smiles, key="emi_mol", height=260)

    if em_smiles and solvent_em and model_emission:
        pair_feats = make_maccs_pair(em_smiles, solvent_em)
        if pair_feats is not None:
            pred = predict_array(model_emission, pair_feats)
            if pred is not None:
                st.success(f"Predicted Emission Max: {pred:.2f} nm")

# =====================================================
# TAB 4 – FRET Analysis (Donor OR Acceptor mode)
# =====================================================
with tab4:
    st.write("Provide **Donor** or **Acceptor** molecule. "
             "The app will predict spectra and find top 5 FRET partners from the dataset.")

    colD, colA = st.columns(2)

    # ----- Donor input -----
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio(
            "Donor Input Method:",
            ("SMILES Input", "Draw Molecule", "Upload File"),
            key="fret_donor_method"
        )
        donor_smiles = ""

        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Donor SMILES", key="fret_donor_smiles_text")
        elif donor_method == "Upload File":
            donor_file = st.file_uploader(
                "Upload Donor (.smi/.mol/.sdf)",
                type=["smi", "mol", "sdf"],
                key="fret_donor_file"
            )
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""
        else:
            donor_smiles = st_ketcher("")
            if donor_smiles:
                st.write(f"Donor SMILES from drawing: `{donor_smiles}`")

        if donor_smiles:
            # Show donor structure
            render_structure(donor_smiles, key="fret_donor_mol", height=240)

    # ----- Acceptor input -----
    with colA:
        st.subheader("Acceptor Molecule")
        acc_method = st.radio(
            "Acceptor Input Method:",
            ("SMILES Input", "Draw Molecule", "Upload File"),
            key="fret_acceptor_method"
        )
        acceptor_smiles = ""

        if acc_method == "SMILES Input":
            acceptor_smiles = st.text_input("Acceptor SMILES", key="fret_acceptor_smiles_text")
        elif acc_method == "Upload File":
            acc_file = st.file_uploader(
                "Upload Acceptor (.smi/.mol/.sdf)",
                type=["smi", "mol", "sdf"],
                key="fret_acceptor_file"
            )
            if acc_file:
                acceptor_smiles = read_molecule_file(acc_file) or ""
        else:
            acceptor_smiles = st_ketcher("")
            if acceptor_smiles:
                st.write(f"Acceptor SMILES from drawing: `{acceptor_smiles}`")

        if acceptor_smiles:
            # Show acceptor structure
            render_structure(acceptor_smiles, key="fret_acceptor_mol", height=240)

    # ----- Validate mode -----
    if not (donor_smiles or acceptor_smiles):
        st.warning("Please provide at least a Donor or an Acceptor molecule.")
    else:
        if model_regression is None or model_emission is None:
            st.error("Absorption / Emission models not loaded. Cannot perform FRET analysis.")
        else:
            # Determine query
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            mode_text = "Donor → Find Acceptors" if is_donor else "Acceptor → Find Donors"
            st.markdown(f"**Mode:** {mode_text}")

            # Predict spectra for query (using water as solvent)
            pair_feats = make_maccs_pair(query_smiles, "O")
            if pair_feats is None:
                st.error("Failed to compute MACCS pair features for query.")
            else:
                with st.spinner("Predicting query spectra..."):
                    pred_abs = predict_array(model_regression, pair_feats)
                    pred_em  = predict_array(model_emission, pair_feats)

                if pred_abs is None or pred_em is None:
                    st.error("Spectral prediction failed.")
                else:
                    if is_donor:
                        st.success(f"Predicted Donor Absorption: {pred_abs:.2f} nm")
                        st.success(f"Predicted Donor Emission: {pred_em:.2f} nm")
                    else:
                        st.success(f"Predicted Acceptor Absorption: {pred_abs:.2f} nm")
                        st.success(f"Predicted Acceptor Emission: {pred_em:.2f} nm")

                    # Load dataset
                    df = load_dataset()
                    if df is None:
                        st.error("Dataset not available for FRET analysis.")
                    else:
                        required = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                        if not required.issubset(df.columns):
                            st.error("Dataset missing required FRET columns.")
                        else:
                            # Use only fluorescent entries, and drop self
                            df_f = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes","true","1"])].copy()
                            df_f = df_f[df_f["Smiles"] != query_smiles]

                            if df_f.empty:
                                st.warning("No fluorescent partners found in dataset.")
                            else:
                                # Compute Δ based on mode
                                if is_donor:
                                    # donor emission vs acceptor absorption
                                    df_f["Δ (nm)"] = (df_f["AbsorptioMax (nm)"] - pred_em).abs()
                                else:
                                    # acceptor absorption vs donor emission
                                    df_f["Δ (nm)"] = (df_f["EmissionMax (nm)"] - pred_abs).abs()

                                top5 = df_f.sort_values("Δ (nm)").head(5).reset_index(drop=True)

                                # Build a compact result table
                                results = pd.DataFrame({
                                    "Dataset SMILES": top5["Smiles"],
                                    "Dataset Absorption (nm)": top5["AbsorptioMax (nm)"],
                                    "Dataset Emission (nm)": top5["EmissionMax (nm)"],
                                    "Δ (nm)": top5["Δ (nm)"],
                                    "Predicted Query Absorption (nm)": [pred_abs] * len(top5),
                                    "Predicted Query Emission (nm)": [pred_em] * len(top5),
                                })

                                st.markdown("**Top 5 FRET Partner Candidates:**")
                                st.dataframe(results, use_container_width=True)

# ----------------- Footer -----------------
st.write("---")
st.caption("FluroML-©PDeshmukh")
