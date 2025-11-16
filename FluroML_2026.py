# FluroML_rdkitjs.py
# Streamlit app using Python RDKit (no rdMolDraw2D) + RDKit.js in-browser rendering (WASM)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import deepchem as dc
from rdkit import Chem   # OK: RDKit core (SMILES parsing, fingerprints), DO NOT import rdkit.Chem.Draw
import urllib.parse
import streamlit.components.v1 as components
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="FluroML - RDKit.js Renderer", layout="wide")

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Error loading model {path}: {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_regression   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")

# -------------------------
# Featurizers (DeepChem)
# -------------------------
morgan_featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
maccs_featurizer = dc.feat.MACCSKeysFingerprint()

def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        features = morgan_featurizer.featurize([mol])[0]
        return np.array(features, dtype=float)
    except Exception as e:
        st.error(f"Morgan featurization error: {e}")
        return None

def smiles_to_maccs(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    try:
        features = maccs_featurizer.featurize([mol])[0]
        return np.array(features, dtype=float)
    except Exception as e:
        st.error(f"MACCS featurization error: {e}")
        return None

# -------------------------
# Prediction helper
# -------------------------
def predict(model, features):
    if model is None or features is None:
        return None
    X = np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        return model.predict(X)[0]
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# -------------------------
# File reader helper
# -------------------------
def read_molecule_file(uploaded_file):
    filename = uploaded_file.name
    data = uploaded_file.getvalue()
    try:
        text = data.decode('utf-8', errors='ignore')
    except Exception:
        text = None
    if filename.lower().endswith('.smi'):
        if not text:
            return None
        lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
        if not lines:
            return None
        return lines[0].split()[0]
    elif filename.lower().endswith('.mol'):
        try:
            mol = Chem.MolFromMolBlock(text)
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None
    elif filename.lower().endswith('.sdf'):
        try:
            blocks = text.split('$$$$')
            mol = Chem.MolFromMolBlock(blocks[0])
            return Chem.MolToSmiles(mol) if mol else None
        except Exception:
            return None
    else:
        return None

# -------------------------
# Dataset loader for FRET
# -------------------------
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return None

# -------------------------
# RDKit.js renderer
# -------------------------
# This HTML tries:
# 1) to load local RDKit files: /RDKit_minimal.js and /rdkit.wasm (best),
# 2) if not present, fall back to CDN URL (may be blocked by CSP).
#
# If you upload RDKit_minimal.js + rdkit.wasm to repo root, the app will load them locally.
#
# IMPORTANT: Adjust CDN URL if you have a preferred RDKit.js build.
CDN_RDKIT_JS = "https://unpkg.com/@rdkit/rdkit/Code/MinimalLib/dist/RDKit_minimal.js"

def render_rdkitjs(smiles: str, key: str):
    """Render SMILES using RDKit.js (WASM) in browser. Tries local files first, then CDN fallback."""
    if not smiles:
        components.html("<div>No structure provided</div>", height=100)
        return

    # encode properly for embedding in JS
    s = smiles.replace("\\", "\\\\").replace('"', '\\"').replace("\n","")
    # HTML: try local RDKit_minimal.js, if load fails then load CDN and initialize
    html = f"""
    <div id="rdkit_container_{key}">Loading RDKit.js renderer...</div>
    <script>
    (function() {{
      function insertScript(src, onload, onerror) {{
        var s = document.createElement('script');
        s.src = src;
        s.onload = onload;
        s.onerror = onerror;
        document.head.appendChild(s);
      }}
      // Try to load local RDKit_minimal.js first
      insertScript("/RDKit_minimal.js", function() {{
        // local rdkit loaded
        initRDKitAndDraw();
      }}, function() {{
        // local not found — try CDN
        insertScript("{CDN_RDKIT_JS}", function() {{
          initRDKitAndDraw();
        }}, function() {{
          document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>Could not load RDKit.js (local or CDN). Please upload RDKit_minimal.js and rdkit.wasm to the repo.</div>";
        }});
      }});
      function initRDKitAndDraw() {{
        try {{
          // RDKit init if needed (some builds expose initRDKitModule)
          if (typeof initRDKitModule !== "undefined") {{
            initRDKitModule().then(function(RDKit) {{
              draw(RDKit);
            }});
          }} else if (typeof RDKit !== "undefined" && RDKit.get_mol) {{
            draw(RDKit);
          }} else {{
            // some builds expose a global 'Module' that gives RDKit
            if (typeof Module !== "undefined" && Module.RDKit) {{
              draw(Module.RDKit);
            }} else {{
              // fallback: try RDKit global again after brief delay
              setTimeout(function() {{
                if (typeof RDKit !== "undefined") draw(RDKit);
                else document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>RDKit loaded but initialization not found.</div>";
              }}, 200);
            }}
          }}
        }} catch (e) {{
          document.getElementById("rdkit_container_{key}").innerHTML = "<div style='color:red'>RDKit init error: " + e + "</div>";
        }}
      }}
      function draw(RDKit) {{
        try {{
          var mol = RDKit.get_mol("{s}");
          if (!mol) {{
            document.getElementById("rdkit_container_{key}").innerHTML = "<div>Invalid SMILES or RDKit could not parse.</div>";
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
    components.html(html, height=360, scrolling=False)

# -------------------------
# UI layout (tabs)
# -------------------------
st.title("FluroML: Molecular Fluorescence Predictor (RDKit.js renderer)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])

# Tab 1: Fluorescence Classification
with tab1:
    st.markdown("## 🧪 Fluorescence Classification")
    input_method = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (external)", "Upload File"), key="fluoro_method")
    smiles = ""
    if input_method == "SMILES Input":
        smiles = st.text_input("Enter a SMILES string:", key="fluorescence_smiles")
    elif input_method == "Upload File":
        file = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fluoro_file")
        if file:
            smiles = read_molecule_file(file) or ""
    else:  # "Draw Molecule (external)" — you can use any drawing tool to paste SMILES
        st.info("Drawing not embedded — paste the exported SMILES here from your editor.")
        smiles = st.text_input("Paste SMILES from your drawing tool:", key="fluorescence_drawn_smiles")

    if smiles:
        # render using RDKit.js in browser
        render_rdkitjs(smiles, key="fluoro1")
        if model_fluorescence:
            fp = smiles_to_morgan(smiles)
            if fp is not None:
                with st.spinner("Predicting fluorescence..."):
                    prediction = predict(model_fluorescence, fp)
                if prediction is None:
                    st.error("Prediction could not be made.")
                else:
                    st.success("Fluorescent" if int(prediction) == 1 else "Non-Fluorescent")
        else:
            st.error("Fluorescence classification model is not loaded.")

# Tab 2: Absorption Max Prediction
with tab2:
    st.markdown("## 🌈 Absorption Max Prediction")
    input_method2 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (external)", "Upload File"), key="abs_method")
    abs_smiles = ""
    if input_method2 == "SMILES Input":
        abs_smiles = st.text_input("Enter Molecule SMILES:", key="absorption_smiles")
    elif input_method2 == "Upload File":
        file2 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="abs_file")
        if file2:
            abs_smiles = read_molecule_file(file2) or ""
    else:
        st.info("Drawing not embedded — paste the exported SMILES here from your editor.")
        abs_smiles = st.text_input("Paste SMILES from your drawing tool:", key="absorption_drawn_smiles")

    solvent = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="absorption_solvent", value="O")
    if abs_smiles and solvent:
        render_rdkitjs(abs_smiles, key="abs1")
        if model_regression:
            desc_smiles = smiles_to_maccs(abs_smiles)
            desc_solvent = smiles_to_maccs(solvent)
            if desc_smiles is not None and desc_solvent is not None:
                features = np.concatenate([desc_smiles, desc_solvent]).reshape(1, -1)
                with st.spinner("Predicting absorption maximum..."):
                    prediction = predict(model_regression, features)
                if prediction is not None:
                    st.write(f"**Predicted Absorption Max:** {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Absorption prediction model is not loaded.")

# Tab 3: Emission Max Prediction
with tab3:
    st.markdown("## 🔦 Emission Max Prediction")
    input_method3 = st.radio("Input Method:", ("SMILES Input", "Draw Molecule (external)", "Upload File"), key="em_method")
    em_smiles = ""
    if input_method3 == "SMILES Input":
        em_smiles = st.text_input("Enter Molecule SMILES:", key="emission_smiles")
    elif input_method3 == "Upload File":
        file3 = st.file_uploader("Upload a molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="em_file")
        if file3:
            em_smiles = read_molecule_file(file3) or ""
    else:
        st.info("Drawing not embedded — paste the exported SMILES here from your editor.")
        em_smiles = st.text_input("Paste SMILES from your drawing tool:", key="emission_drawn_smiles")

    solvent_em = st.text_input("Enter Solvent SMILES (e.g., 'O' for water):", key="emission_solvent", value="O")
    if em_smiles and solvent_em:
        render_rdkitjs(em_smiles, key="em1")
        if model_emission:
            desc_smiles = smiles_to_maccs(em_smiles)
            desc_solvent = smiles_to_maccs(solvent_em)
            if desc_smiles is not None and desc_solvent is not None:
                features = np.concatenate([desc_smiles, desc_solvent]).reshape(1, -1)
                with st.spinner("Predicting emission maximum..."):
                    prediction = predict(model_emission, features)
                if prediction is not None:
                    st.write(f"**Predicted Emission Max:** {prediction:.2f} nm")
                else:
                    st.error("Prediction could not be made.")
        else:
            st.error("Emission prediction model is not loaded.")

# Tab 4: FRET Pair Analysis
with tab4:
    st.markdown("## 🔬 FRET Pair Analysis (RDKit.js rendering)")
    input_method4 = st.radio("Donor Input Method:", ("SMILES Input", "Draw Molecule (external)", "Upload File"), key="fret_method")
    donor_smiles = ""
    if input_method4 == "SMILES Input":
        donor_smiles = st.text_input("Enter Donor Molecule SMILES:", key="fret_donor_smiles")
    elif input_method4 == "Upload File":
        file4 = st.file_uploader("Upload a donor molecule file (.smi, .mol, .sdf):", type=["smi", "mol", "sdf"], key="fret_file")
        if file4:
            donor_smiles = read_molecule_file(file4) or ""
    else:
        st.info("Drawing not embedded — paste the exported SMILES here from your editor.")
        donor_smiles = st.text_input("Paste SMILES from your drawing tool:", key="fret_donor_drawn")

    if donor_smiles:
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
                    desc = smiles_to_maccs(donor_smiles)
                    solvent_desc = smiles_to_maccs("O")
                    if desc is not None and solvent_desc is not None:
                        features = np.concatenate([desc, solvent_desc]).reshape(1, -1)
                        with st.spinner("Searching for optimal FRET acceptor..."):
                            donor_emission = predict(model_emission, features)
                            df_candidates = df[df['Fluorescent labeling'].astype(str).str.lower().isin(["yes", "true", "1"])].copy()
                            df_candidates = df_candidates[df_candidates['AbsorptioMax (nm)'].notna()]
                            df_candidates = df_candidates[df_candidates['Smiles'] != donor_smiles]
                            if df_candidates.empty or donor_emission is None:
                                st.error("No suitable acceptor candidates found (or emission prediction failed).")
                            else:
                                df_candidates['abs_diff'] = (df_candidates['AbsorptioMax (nm)'] - donor_emission).abs()
                                best_per_smiles = df_candidates.loc[df_candidates.groupby('Smiles')['abs_diff'].idxmin()]
                                best_per_smiles = best_per_smiles.sort_values('abs_diff').reset_index(drop=True)
                                top_match = best_per_smiles.iloc[0]
                                best_smiles = top_match['Smiles']
                                best_abs = float(top_match['AbsorptioMax (nm)'])
                                best_em = float(top_match['EmissionMax (nm)']) if pd.notna(top_match['EmissionMax (nm)']) else None
                                fret_eff = (best_abs / (donor_emission + best_abs)) * 100
                                col1, col2 = st.columns(2)
                                with col1:
                                    render_rdkitjs(donor_smiles, key="fret_donor")
                                    st.write(f"**Predicted Donor Emission Max:** {donor_emission:.2f} nm")
                                with col2:
                                    render_rdkitjs(best_smiles, key="fret_acceptor")
                                    st.write(f"**Acceptor Absorption Max:** {best_abs:.2f} nm")
                                    if best_em is not None:
                                        st.write(f"**Acceptor Emission Max:** {best_em:.2f} nm")
                                    st.write(f"**FRET Efficiency:** {fret_eff:.2f}%")
                                top_n = 5
                                top_candidates = best_per_smiles.head(top_n)[['Smiles', 'AbsorptioMax (nm)', 'EmissionMax (nm)', 'abs_diff']].copy()
                                top_candidates.rename(columns={
                                    'Smiles': 'Acceptor SMILES',
                                    'AbsorptioMax (nm)': 'Absorption (nm)',
                                    'EmissionMax (nm)': 'Emission (nm)',
                                    'abs_diff': 'Δ (nm)'
                                }, inplace=True)
                                top_candidates['Absorption (nm)'] = top_candidates['Absorption (nm)'].map(lambda x: f"{x:.2f}")
                                top_candidates['Emission (nm)'] = top_candidates['Emission (nm)'].map(lambda x: f"{x:.2f}" if str(x) != 'nan' else "N/A")
                                top_candidates['Δ (nm)'] = top_candidates['Δ (nm)'].map(lambda x: f"{x:.2f}")
                                st.markdown("**Top 5 Closest Matches:**")
                                st.table(top_candidates.reset_index(drop=True))

# Footer
st.write("---")
st.caption("FluroML-©PDeshmukh")
