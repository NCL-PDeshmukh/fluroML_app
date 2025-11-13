# fluroml_jsme_with_donor_absorption.py
import streamlit as st
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc
import streamlit.components.v1 as components
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt

# --- JSME Editor helper (Full/default toolbar) ---
def jsme_editor(key: str, width: int = 520, height: int = 360, initial_smiles: str = "") -> str:
    """
    Render a JSME editor with the default (full) toolbar.
    On clicking 'Export SMILES' the page will reload with a query param:
       ?jsme_<key>=<URL-encoded smiles>
    This function returns the SMILES string found in the query param (if any).
    """
    param_name = f"jsme_{key}"
    query_params = st.experimental_get_query_params()
    # If there's a SMILES present in query params for this editor, return it
    if param_name in query_params:
        smiles_val = query_params[param_name][0]
        return smiles_val

    # Otherwise render the editor HTML (which will allow user to export SMILES)
    jsme_js_url = "https://jsme-editor.github.io/dist/jsme.nocache.js"
    # ensure initial_smiles is properly encoded for use in JS
    initial_enc = urllib.parse.quote(initial_smiles or "")
    html = f"""
    <div id="jsme_container_{key}" style="border:1px solid #ddd; width:{width}px; height:{height}px;"></div>
    <div style="margin-top:6px;">
      <button id="export_{key}" style="margin-right:8px;">Export SMILES</button>
      <button id="clear_{key}">Clear Editor</button>
      <span style="margin-left:10px;color:#555;font-size:0.9rem;">(Use Export to send SMILES back to the app)</span>
    </div>

    <script src="{jsme_js_url}"></script>
    <script>
      (function() {{
        function initJSME() {{
          try {{
            var jsme = new JSApplet.JSME("jsme_container_{key}", "{width-20}px", "{height-70}px");
            var initial = "{initial_enc}";
            if (initial && initial.length > 0) {{
              try {{ jsme.setSmiles(decodeURIComponent(initial)); }} catch(e){{}}
            }}

            document.getElementById("export_{key}").onclick = function() {{
              try {{
                var smi = jsme.smiles();
                if(!smi) {{
                  alert("No structure in the editor to export.");
                  return;
                }}
                var enc = encodeURIComponent(smi);
                var url = new URL(window.location.href);
                url.searchParams.set("jsme_{key}", enc);
                window.location.href = url.toString();
              }} catch (err) {{
                alert("Failed to export SMILES: " + err);
              }}
            }};

            document.getElementById("clear_{key}").onclick = function() {{
              try {{
                jsme.setSmiles("");
              }} catch(e){{ console.log(e); }}
            }};
          }} catch (e) {{
            setTimeout(initJSME, 200);
          }}
        }}
        initJSME();
      }})();
    </script>
    """
    components.html(html, height=height + 80)
    return ""  # empty until exported (page reload will provide smi in next run)


# --- Streamlit App configuration ---
st.set_page_config(
    page_title="FluroML - Molecular Prediction (JSME)",
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
model_regression   = load_model("new_best_regressor_compatible.joblib")  # absorption model
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# Helper functions
def smiles_to_morgan(smiles: str):
    """Convert SMILES to Morgan fingerprint vector using DeepChem CircularFingerprint."""
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
    return pd.DataFrame(features)

def predict(model, features):
    """Make a prediction using a trained model and feature vector."""
    if model is None:
        return None
    import numpy as _np
    X = features.values if isinstance(features, pd.DataFrame) else _np.array(features)
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
    if filename.lower().endswith('.smi'):
        content = data.decode('utf-8', errors='ignore')
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            st.error("SMI file is empty or invalid.")
            return None
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

# Notify user about drawing method
st.info("This app uses the JSME 2D editor for structure drawing. Use 'Export SMILES' to send the structure to the app.")

# App layout
st.title("FluroML: Molecular Fluorescence Predictor (JSME Editor)")
tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# ---------- TAB 1 ----------
with tab1:
    st.markdown("## 🧪 Fluorescence Classification")
    input_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="fluoro_method")
    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="fluorescence_smiles")
    elif input_method == "Upload File":
        file = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fluoro_file")
        if file:
            smiles = read_molecule_file(file) or ""
    else:  # Draw Molecule with JSME
        jsmi = jsme_editor("fluoro", width=520, height=360, initial_smiles="")
        if jsmi:
            smiles = urllib.parse.unquote(jsmi)
            st.success("Structure exported from JSME.")
            st.write(f"**SMILES from drawing:** `{smiles}`")

    if smiles:
        if model_fluorescence:
            features = smiles_to_morgan(smiles)
            if features is not None:
                with st.spinner("Predicting fluorescence..."):
                    prediction = predict(model_fluorescence, features)
                st.write("Molecule SMILES:", smiles)
                if prediction is None:
                    st.error("Prediction could not be made.")
                else:
                    try:
                        label_int = int(prediction)
                        st.success("Fluorescent" if label_int == 1 else "Non-Fluorescent")
                    except Exception:
                        st.write(f"Prediction: {prediction}")
        else:
            st.error("Fluorescence classification model is not loaded.")

# ---------- TAB 2 ----------
with tab2:
    st.markdown("## 🌈 Absorption Max Prediction")
    input_method2 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="abs_method")
    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="absorption_smiles")
    elif input_method2 == "Upload File":
        file2 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="abs_file")
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        jsmi2 = jsme_editor("abs", width=520, height=360, initial_smiles="")
        if jsmi2:
            abs_smiles = urllib.parse.unquote(jsmi2)
            st.success("Structure exported from JSME.")
            st.write(f"**SMILES from drawing:** `{abs_smiles}`")

    solvent = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="absorption_solvent")
    if abs_smiles and solvent:
        if model_regression:
            desc_smiles = smiles_to_descriptors(abs_smiles)
            desc_solvent = smiles_to_descriptors(solvent)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting absorption maximum..."):
                    prediction = predict(model_regression, features)
                st.write("Molecule SMILES:", abs_smiles)
                if prediction is not None:
                    try:
                        st.write(f"**Predicted Absorption Max:** {float(prediction):.2f} nm")
                    except Exception:
                        st.write(f"**Predicted Absorption Max:** {prediction}")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Absorption prediction model is not loaded.")

# ---------- TAB 3 ----------
with tab3:
    st.markdown("## 🔦 Emission Max Prediction")
    input_method3 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="em_method")
    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emission_smiles")
    elif input_method3 == "Upload File":
        file3 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="em_file")
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        jsmi3 = jsme_editor("em", width=520, height=360, initial_smiles="")
        if jsmi3:
            em_smiles = urllib.parse.unquote(jsmi3)
            st.success("Structure exported from JSME.")
            st.write(f"**SMILES from drawing:** `{em_smiles}`")

    solvent_em = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="emission_solvent")
    if em_smiles and solvent_em:
        if model_emission:
            desc_smiles = smiles_to_descriptors(em_smiles)
            desc_solvent = smiles_to_descriptors(solvent_em)
            if desc_smiles is not None and desc_solvent is not None:
                features = pd.concat([desc_smiles, desc_solvent], axis=1)
                with st.spinner("Predicting emission maximum..."):
                    prediction = predict(model_emission, features)
                st.write("Molecule SMILES:", em_smiles)
                if prediction is not None:
                    try:
                        st.write(f"**Predicted Emission Max:** {float(prediction):.2f} nm")
                    except Exception:
                        st.write(f"**Predicted Emission Max:** {prediction}")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Emission prediction model is not loaded.")

# ---------- TAB 4: FRET ----------
with tab4:
    st.markdown("## 🔬 FRET Pair Analysis")
    st.markdown("Provide **Donor** or **Acceptor** molecule for FRET compatibility and dataset-based spectral overlap search.")

    colD, colA = st.columns(2)

    # Donor input
    with colD:
        st.subheader("Donor Molecule")
        donor_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="fret_donor_method")
        donor_smiles = ""
        if donor_method == "SMILES Input":
            donor_smiles = st.text_input("Enter Donor SMILES:", key="fret_donor_smiles")
        elif donor_method == "Upload File":
            donor_file = st.file_uploader("Upload Donor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_donor_file")
            if donor_file:
                donor_smiles = read_molecule_file(donor_file) or ""
        else:
            jsmi_d = jsme_editor("fret_donor", width=480, height=320, initial_smiles="")
            if jsmi_d:
                donor_smiles = urllib.parse.unquote(jsmi_d)
                st.success("Structure exported from JSME.")
                st.write(f"**Donor SMILES:** `{donor_smiles}`")

    # Acceptor input
    with colA:
        st.subheader("Acceptor Molecule")
        acceptor_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (JSME)", "Upload File"), key="fret_acceptor_method")
        acceptor_smiles = ""
        if acceptor_method == "SMILES Input":
            acceptor_smiles = st.text_input("Enter Acceptor SMILES:", key="fret_acceptor_smiles")
        elif acceptor_method == "Upload File":
            acceptor_file = st.file_uploader("Upload Acceptor (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_acceptor_file")
            if acceptor_file:
                acceptor_smiles = read_molecule_file(acceptor_file) or ""
        else:
            jsmi_a = jsme_editor("fret_acceptor", width=480, height=320, initial_smiles="")
            if jsmi_a:
                acceptor_smiles = urllib.parse.unquote(jsmi_a)
                st.success("Structure exported from JSME.")
                st.write(f"**Acceptor SMILES:** `{acceptor_smiles}`")

    # FRET analysis (display donor absorption + existing Δ logic kept)
    if donor_smiles or acceptor_smiles:
        if model_emission is None or model_regression is None:
            st.error("Required models (emission or absorption) not loaded. Cannot perform FRET analysis.")
        else:
            is_donor = bool(donor_smiles)
            query_smiles = donor_smiles if is_donor else acceptor_smiles
            st.markdown(f"### 🔹 Mode: {'Donor → Find Acceptors' if is_donor else 'Acceptor → Find Donors'}")

            desc = smiles_to_descriptors(query_smiles)
            solvent_desc = smiles_to_descriptors("O")  # assume water

            if desc is not None and solvent_desc is not None:
                with st.spinner("Predicting spectral property..."):
                    features = pd.concat([desc, solvent_desc], axis=1)
                    if is_donor:
                        # Predict donor emission
                        query_em = predict(model_emission, features)
                        # Predict donor absorption (display only)
                        try:
                            donor_abs = predict(model_regression, features)
                        except Exception:
                            donor_abs = None

                        st.write("### 🔷 Donor Predicted Properties")
                        if donor_abs is not None:
                            try:
                                st.write(f"**Predicted Donor Absorption Max:** {float(donor_abs):.2f} nm")
                            except Exception:
                                st.write(f"**Predicted Donor Absorption Max:** {donor_abs}")
                        else:
                            st.info("Donor absorption prediction unavailable.")

                        try:
                            st.write(f"**Predicted Donor Emission Max:** {float(query_em):.2f} nm")
                        except Exception:
                            st.write(f"**Predicted Donor Emission Max:** {query_em}")
                    else:
                        query_abs = predict(model_regression, features)
                        try:
                            st.write(f"**Predicted Acceptor Absorption Max:** {float(query_abs):.2f} nm")
                        except Exception:
                            st.write(f"**Predicted Acceptor Absorption Max:** {query_abs}")

                # ---- Dataset-based FRET Partner Search ----
                st.markdown("### 🔍 Searching for best matching FRET partners...")
                df = load_dataset()

                if df is not None and {"Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Fluorescent labeling"}.issubset(df.columns):
                    df_fluoro = df[df['Fluorescent labeling'].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                    df_fluoro = df_fluoro[df_fluoro['Smiles'] != query_smiles]

                    if is_donor:
                        # Δ remains |Acceptor Absorption - Donor Emission| (display only donor absorption)
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

                    # Optional visualization: overlap plots (unchanged)
                    if is_donor:
                        try:
                            donor_em_val = float(query_em)
                        except Exception:
                            donor_em_val = None
                        for idx, row in top_candidates.iterrows():
                            acceptor_abs = float(row["Absorption (nm)"])
                            wavelength = np.linspace(300, 800, 1000)
                            if donor_em_val is None:
                                # skip plotting if donor_em not numeric
                                continue
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em_val) / 20) ** 2)
                            acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs) / 25) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label="Donor Emission", lw=2)
                            ax.plot(wavelength, acceptor_curve, label=f"Acceptor Absorption ({row['Absorption (nm)']:.1f} nm)", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em_val - acceptor_abs):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                    else:
                        acceptor_abs_val = float(query_abs)
                        for idx, row in top_candidates.iterrows():
                            donor_em = float(row["Emission (nm)"])
                            wavelength = np.linspace(300, 800, 1000)
                            donor_curve = np.exp(-0.5 * ((wavelength - donor_em) / 20) ** 2)
                            acceptor_curve = np.exp(-0.5 * ((wavelength - acceptor_abs_val) / 25) ** 2)
                            overlap_area = np.trapz(np.minimum(donor_curve, acceptor_curve), wavelength)
                            overlap_pct = overlap_area / np.trapz(donor_curve, wavelength) * 100
                            fig, ax = plt.subplots(figsize=(6, 3))
                            ax.plot(wavelength, donor_curve, label=f"Donor Emission ({row['Emission (nm)']:.1f} nm)", lw=2)
                            ax.plot(wavelength, acceptor_curve, label="Acceptor Absorption", lw=2)
                            ax.fill_between(wavelength, np.minimum(donor_curve, acceptor_curve), alpha=0.3)
                            ax.set_xlabel("Wavelength (nm)")
                            ax.set_ylabel("Intensity")
                            ax.set_title(f"Overlap ≈ {overlap_pct:.1f}% | Δλ={abs(donor_em - acceptor_abs_val):.1f} nm")
                            ax.legend()
                            st.pyplot(fig)
                else:
                    st.info("Dataset not available or missing necessary columns for FRET partner search.")
    else:
        st.warning("Please provide at least a Donor or an Acceptor molecule to begin FRET analysis.")

# Footer
st.write("---")
st.caption("FluroML-©PDeshmukh")
