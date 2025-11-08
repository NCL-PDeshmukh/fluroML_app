import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# ---- RDKit (safe import) ----
_rdkit_import_error = None
try:
    from rdkit import Chem
    from rdkit.Chem import Draw, AllChem, MACCSkeys
except Exception as e:
    Chem = Draw = AllChem = MACCSkeys = None
    _rdkit_import_error = str(e)

# ---- Optional drawing widget ----
try:
    from streamlit_ketcher import st_ketcher
except ImportError:
    def st_ketcher(*args, **kwargs):
        st.warning("streamlit-ketcher is not installed. Proceeding without the drawing widget.")
        return ""

# ---------------- UI CONFIG ----------------
st.set_page_config(
    page_title="FluroML - Molecular Prediction",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("FluroML: Molecular Fluorescence Predictor")

if Chem is None:
    st.error(
        "RDKit is not available. Please ensure `rdkit==2023.9.5` is in requirements.txt.\n\n"
        f"Import error: {_rdkit_import_error or 'unknown'}"
    )

# ---------------- UTIL ----------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model from {path}: {e}")
        return None

# Load models
model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

def _require_rdkit():
    if Chem is None:
        st.stop()

def smiles_to_morgan(smiles: str, radius: int = 3, n_bits: int = 1024):
    """Morgan fingerprint as a numpy vector."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    try:
        bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
        arr = np.zeros((n_bits,), dtype=int)
        # Convert ExplicitBitVect -> numpy
        for i in range(n_bits):
            arr[i] = int(bv.GetBit(i))
        return arr
    except Exception as e:
        st.error(f"Failed to compute Morgan fingerprint: {e}")
        return None

def smiles_to_maccs_df(smiles: str) -> pd.DataFrame | None:
    """MACCS keys as a single-row DataFrame (167 bits)."""
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    try:
        fp = MACCSkeys.GenMACCSKeys(mol)  # 167 bits
        n_bits = fp.GetNumBits()
        arr = np.array([int(fp.GetBit(i)) for i in range(n_bits)], dtype=int)
        # Make stable column names: MACCS_0..MACCS_166
        cols = [f"MACCS_{i}" for i in range(n_bits)]
        return pd.DataFrame([arr], columns=cols)
    except Exception as e:
        st.error(f"Failed to compute MACCS keys: {e}")
        return None

def draw_molecule(smiles: str):
    _require_rdkit()
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return Draw.MolToImage(mol, size=(300, 300))
    return None

def predict(model, features):
    if model is None:
        return None
    X = features.values if isinstance(features, pd.DataFrame) else np.asarray(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        pred = model.predict(X)
        # Some sklearn regressors return array([value])
        if hasattr(pred, "__len__") and len(pred) == 1:
            return float(pred[0])
        return pred
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

def read_molecule_file(uploaded_file):
    _require_rdkit()
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    try:
        if filename.lower().endswith(".smi"):
            content = data.decode("utf-8", errors="ignore")
            lines = [l.strip() for l in content.splitlines() if l.strip()]
            if not lines:
                st.error("SMI file is empty or invalid.")
                return None
            return lines[0].split()[0]
        elif filename.lower().endswith(".mol"):
            mol_block = data.decode("utf-8", errors="ignore")
            mol = Chem.MolFromMolBlock(mol_block)
            if mol is None:
                st.error("Failed to parse MOL file.")
                return None
            return Chem.MolToSmiles(mol)
        elif filename.lower().endswith(".sdf"):
            content = data.decode("utf-8", errors="ignore")
            entries = [e for e in content.split("$$$$") if e.strip()]
            if not entries:
                st.error("No molecules found in SDF.")
                return None
            mol = Chem.MolFromMolBlock(entries[0])
            if mol is None:
                st.error("Failed to parse first molecule in SDF.")
                return None
            if len(entries) > 1:
                st.info("Multiple molecules in SDF; using the first one.")
            return Chem.MolToSmiles(mol)
        else:
            st.error("Unsupported file type (use .smi, .mol, .sdf).")
            return None
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# ---------------- LAYOUT ----------------
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis",
])

# ------- TAB 1 -------
with tab1:
    st.markdown("## 🧪 Fluorescence Classification")
    method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="fluoro_method")
    smiles = ""
    if method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="fluorescence_smiles")
    elif method == "Upload File":
        f = st.file_uploader("Upload (.smi, .mol, .sdf)", type=["smi", "mol", "sdf"], key="fluoro_file")
        if f:
            smiles = read_molecule_file(f) or ""
    else:
        smiles = st_ketcher("") or ""

    if smiles:
        if model_fluorescence is None:
            st.error("Fluorescence model not loaded.")
        else:
            feats = smiles_to_morgan(smiles)
            if feats is not None:
                with st.spinner("Predicting fluorescence..."):
                    y = predict(model_fluorescence, feats)
                if Chem:
                    st.image(draw_molecule(smiles), caption="Molecule Structure")
                if y is None:
                    st.error("Prediction failed.")
                else:
                    try:
                        label = int(round(float(y)))
                        st.success("Fluorescent" if label == 1 else "Non-Fluorescent")
                    except Exception:
                        st.write(f"Prediction: {y}")

# ------- TAB 2 -------
with tab2:
    st.markdown("## 🌈 Absorption Max Prediction")
    method2 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="abs_method")
    abs_smiles = ""
    if method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="absorption_smiles")
    elif method2 == "Upload File":
        f2 = st.file_uploader("Upload (.smi, .mol, .sdf)", type=["smi", "mol", "sdf"], key="abs_file")
        if f2:
            abs_smiles = read_molecule_file(f2) or ""
    else:
        abs_smiles = st_ketcher("") or ""

    solvent = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", value="O", key="absorption_solvent")

    if abs_smiles and solvent:
        if model_regression is None:
            st.error("Absorption model not loaded.")
        else:
            d_mol = smiles_to_maccs_df(abs_smiles)
            d_sol = smiles_to_maccs_df(solvent)
            if d_mol is not None and d_sol is not None:
                X = pd.concat([d_mol, d_sol], axis=1)
                with st.spinner("Predicting absorption maximum..."):
                    y = predict(model_regression, X)
                if Chem:
                    st.image(draw_molecule(abs_smiles), caption="Molecule Structure")
                if y is None:
                    st.error("Prediction failed.")
                else:
                    st.write(f"**Predicted Absorption Max:** {float(y):.2f} nm")

# ------- TAB 3 -------
with tab3:
    st.markdown("## 🔦 Emission Max Prediction")
    method3 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule", "Upload File"), key="em_method")
    em_smiles = ""
    if method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emission_smiles")
    elif method3 == "Upload File":
        f3 = st.file_uploader("Upload (.smi, .mol, .sdf)", type=["smi", "mol", "sdf"], key="em_file")
        if f3:
            em_smiles = read_molecule_file(f3) or ""
    else:
        em_smiles = st_ketcher("") or ""

    solvent_em = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", value="O", key="emission_solvent")

    if em_smiles and solvent_em:
        if model_emission is None:
            st.error("Emission model not loaded.")
        else:
            d_mol = smiles_to_maccs_df(em_smiles)
            d_sol = smiles_to_maccs_df(solvent_em)
            if d_mol is not None and d_sol is not None:
                X = pd.concat([d_mol, d_sol], axis=1)
                with st.spinner("Predicting emission maximum..."):
                    y = predict(model_emission, X)
                if Chem:
                    st.image(draw_molecule(em_smiles), caption="Molecule Structure")
                if y is None:
                    st.error("Prediction failed.")
                else:
                    st.write(f"**Predicted Emission Max:** {float(y):.2f} nm")

# ------- TAB 4 (FRET) -------
with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** to search dataset for best spectral matches.")

    colD, colA = st.columns(2)
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
            donor_smiles = st_ketcher("") or ""

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
            acceptor_smiles = st_ketcher("") or ""

    if donor_smiles or acceptor_smiles:
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded.")
        else:
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            d_query = smiles_to_maccs_df(query_smiles)
            d_water = smiles_to_maccs_df("O")  # assume water
            if d_query is not None and d_water is not None:
                X = pd.concat([d_query, d_water], axis=1)
                if is_donor:
                    query_em = predict(model_emission, X)
                    if Chem:
                        st.image(draw_molecule(query_smiles), caption="Donor Molecule")
                    st.write(f"**Predicted Donor Emission Max:** {float(query_em):.2f} nm" if query_em is not None else "Emission prediction failed.")
                else:
                    query_abs = predict(model_regression, X)
                    if Chem:
                        st.image(draw_molecule(query_smiles), caption="Acceptor Molecule")
                    st.write(f"**Predicted Acceptor Absorption Max:** {float(query_abs):.2f} nm" if query_abs is not None else "Absorption prediction failed.")

                st.markdown("### 🔍 Searching for best matching FRET partners...")
                df = load_dataset()
                required = {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}
                if df is not None and required.issubset(df.columns):
                    df_f = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                    df_f = df_f[df_f["Smiles"] != query_smiles]

                    if is_donor and query_em is not None:
                        df_f["Δ (nm)"] = (df_f["AbsorptioMax (nm)"] - float(query_em)).abs()
                        top = df_f.sort_values("Δ (nm)").head(5).rename(columns={
                            "Smiles": "Acceptor SMILES",
                            "AbsorptioMax (nm)": "Absorption (nm)",
                            "EmissionMax (nm)": "Emission (nm)",
                        })
                        st.markdown("### 🧩 Top 5 FRET Acceptor Candidates")
                    elif (not is_donor) and query_abs is not None:
                        df_f["Δ (nm)"] = (df_f["EmissionMax (nm)"] - float(query_abs)).abs()
                        top = df_f.sort_values("Δ (nm)").head(5).rename(columns={
                            "Smiles": "Donor SMILES",
                            "AbsorptioMax (nm)": "Absorption (nm)",
                            "EmissionMax (nm)": "Emission (nm)",
                        })
                        st.markdown("### 🧩 Top 5 FRET Donor Candidates")
                    else:
                        top = None

                    if top is not None:
                        st.table(top.reset_index(drop=True))

                        # Simple overlap visualization (Gaussian toy model)
                        def plot_overlap(em_peak, abs_peak, title_note=""):
                            wl = np.linspace(300, 800, 1000)
                            donor_curve = np.exp(-0.5 * ((wl - em_peak) / 20) ** 2)
                            accept_curve = np.exp(-0.5 * ((wl - abs_peak) / 25) ** 2)
                            area = np.trapz(np.minimum(donor_curve, accept_curve), wl)
                            pct = area / np.trapz(donor_curve, wl) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wl, donor_curve, label="Donor Emission", lw=2)
                            ax.plot(wl, accept_curve, label="Acceptor Absorption", lw=2)
                            ax.fill_between(wl, np.minimum(donor_curve, accept_curve), alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {pct:.1f}% {title_note}")
                            ax.legend()
                            st.pyplot(fig)

                        if is_donor and query_em is not None:
                            em_peak = float(query_em)
                            for _, row in top.iterrows():
                                plot_overlap(em_peak, float(row["Absorption (nm)"]),
                                             title_note=f"| Δλ={abs(em_peak - float(row['Absorption (nm)'])):.1f} nm")
                        elif (not is_donor) and query_abs is not None:
                            abs_peak = float(query_abs)
                            for _, row in top.iterrows():
                                plot_overlap(float(row["Emission (nm)"]), abs_peak,
                                             title_note=f"| Δλ={abs(float(row['Emission (nm)']) - abs_peak):.1f} nm")
                else:
                    st.info("Dataset missing or lacks required columns: "
                            "`Smiles`, `AbsorptioMax (nm)`, `EmissionMax (nm)`, `Fluorescent labeling`.")
    else:
        st.warning("Provide at least a Donor or an Acceptor molecule to begin FRET analysis.")

# Footer
st.write("---")
st.caption("FluroML-©PDeshmukh")

