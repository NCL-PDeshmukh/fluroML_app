import streamlit as st
from rdkit import Chem
import joblib
import pandas as pd
import deepchem as dc
import streamlit.components.v1 as components
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt

# =====================================================================================
# ✔ 2D Molecule Viewer (MolView – ALWAYS works on Streamlit Cloud)
# =====================================================================================
def show_mol(smiles: str, width=450, height=350):
    """Display a 2D molecule using MolView iframe (cloud safe, no RDKit drawing)."""
    if not smiles:
        return
    encoded = urllib.parse.quote(smiles)
    html = f"""
    <iframe 
        src="https://molview.org/?smiles={encoded}"
        width="{width}" 
        height="{height}" 
        style="border:0;">
    </iframe>
    """
    components.html(html, height=height + 10)


# =====================================================================================
# ✔ JSME Editor (full default toolbar, no banners)
# =====================================================================================
def jsme_editor(key: str, width: int = 520, height: int = 360, initial_smiles: str = "") -> str:
    param_name = f"jsme_{key}"
    params = st.experimental_get_query_params()

    if param_name in params:
        return params[param_name][0]

    jsme_js_url = "https://jsme-editor.github.io/dist/jsme.nocache.js"
    init_enc = urllib.parse.quote(initial_smiles or "")

    html = f"""
    <div id="jsme_container_{key}" style="border:1px solid #ddd; width:{width}px; height:{height}px;"></div>
    <div style="margin-top:6px;">
      <button id="export_{key}" style="margin-right:8px;">Export SMILES</button>
      <button id="clear_{key}">Clear</button>
    </div>

    <script src="{jsme_js_url}"></script>
    <script>
        function initEditor() {{
            try {{
                var jsme = new JSApplet.JSME("jsme_container_{key}", "{width-20}px", "{height-70}px");
                var init = "{init_enc}";
                if(init) {{
                    try {{ jsme.setSmiles(decodeURIComponent(init)); }} catch(e){{}}
                }}

                document.getElementById("export_{key}").onclick = function() {{
                    var smi = jsme.smiles();
                    if(!smi) {{
                        alert("Draw a structure first.");
                        return;
                    }}
                    var enc = encodeURIComponent(smi);
                    var url = new URL(window.location.href);
                    url.searchParams.set("{param_name}", enc);
                    window.location.href = url.toString();
                }};

                document.getElementById("clear_{key}").onclick = function() {{
                    jsme.setSmiles("");
                }};
            }} catch(e) {{
                setTimeout(initEditor,200);
            }}
        }}
        initEditor();
    </script>
    """
    components.html(html, height=height + 80)
    return ""


# =====================================================================================
# ✔ Streamlit Page Config
# =====================================================================================
st.set_page_config(
    page_title="FluroML - Molecular Prediction (JSME + MolView)",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================================
# ✔ Load Models
# =====================================================================================
@st.cache_resource
def load_model(path: str):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Model load error ({path}): {e}")
        return None

model_fluorescence = load_model("best_classifier_compatible.joblib")
model_absorption   = load_model("new_best_regressor_compatible.joblib")
model_emission     = load_model("best_regressor_emission_compatible.joblib")


# =====================================================================================
# ✔ Featurization helpers
# =====================================================================================
def smiles_to_morgan(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    featurizer = dc.feat.CircularFingerprint(radius=3, size=1024)
    return featurizer.featurize([mol])[0]


def smiles_to_maccs_df(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    featurizer = dc.feat.MACCSKeysFingerprint()
    data = featurizer.featurize([mol])
    return pd.DataFrame(data)


def predict(model, features):
    if model is None:
        return None
    import numpy as np
    X = features.values if isinstance(features, pd.DataFrame) else np.array(features)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    out = model.predict(X)
    return out[0]


# =====================================================================================
# ✔ File Reader
# =====================================================================================
def read_molecule_file(f):
    name = f.name.lower()
    data = f.getvalue().decode("utf-8", errors="ignore")

    if name.endswith(".smi"):
        return data.splitlines()[0].split()[0]

    if name.endswith(".mol") or name.endswith(".sdf"):
        mol = Chem.MolFromMolBlock(data)
        return Chem.MolToSmiles(mol) if mol else None

    st.error("Unsupported file type.")
    return None


# =====================================================================================
# ✔ Dataset For FRET
# =====================================================================================
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(f"Dataset error: {e}")
        return None


# =====================================================================================
# ✔ Application Tabs
# =====================================================================================
st.title("FluroML: Molecular Fluorescence Predictor (JSME + 2D Structures)")


tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis"
])


# =====================================================================================
# TAB 1 — Fluorescence Classification
# =====================================================================================
with tab1:
    st.header("🧪 Fluorescence Classification")

    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"])
    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Enter SMILES:")

    elif method == "Upload File":
        f = st.file_uploader("Upload .smi / .mol / .sdf", type=["smi", "mol", "sdf"])
        if f:
            smi = read_molecule_file(f)

    else:
        js = jsme_editor("fluoro")
        if js:
            smi = urllib.parse.unquote(js)
            st.success("Molecule exported from JSME.")

    if smi:
        st.subheader("Structure")
        show_mol(smi)

        features = smiles_to_morgan(smi)
        if features is not None:
            pred = predict(model_fluorescence, features)
            st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")


# =====================================================================================
# TAB 2 — Absorption Max Prediction
# =====================================================================================
with tab2:
    st.header("🌈 Absorption Max Prediction")

    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"])
    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Molecule SMILES:")

    elif method == "Upload File":
        f = st.file_uploader("Upload .smi / .mol / .sdf", type=["smi", "mol", "sdf"])
        if f:
            smi = read_molecule_file(f)
    else:
        js = jsme_editor("absorb")
        if js:
            smi = urllib.parse.unquote(js)

    solvent = st.text_input("Solvent SMILES (e.g., O for water):")

    if smi:
        st.subheader("Structure")
        show_mol(smi)

    if smi and solvent:
        feat = smiles_to_maccs_df(smi)
        solv = smiles_to_maccs_df(solvent)

        if feat is not None and solv is not None:
            X = pd.concat([feat, solv], axis=1)
            pred = predict(model_absorption, X)
            st.info(f"Predicted Absorption Max: **{float(pred):.2f} nm**")


# =====================================================================================
# TAB 3 — Emission Max Prediction
# =====================================================================================
with tab3:
    st.header("🔦 Emission Max Prediction")

    method = st.radio("Input Method:", ["SMILES Input", "Draw Molecule (JSME)", "Upload File"])
    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Molecule SMILES:")

    elif method == "Upload File":
        f = st.file_uploader("Upload .smi / .mol / .sdf", type=["smi", "mol", "sdf"])
        if f:
            smi = read_molecule_file(f)
    else:
        js = jsme_editor("emission")
        if js:
            smi = urllib.parse.unquote(js)

    solvent = st.text_input("Solvent SMILES:")

    if smi:
        st.subheader("Structure")
        show_mol(smi)

    if smi and solvent:
        feat = smiles_to_maccs_df(smi)
        solv = smiles_to_maccs_df(solvent)
        X = pd.concat([feat, solv], axis=1)

        pred = predict(model_emission, X)
        st.info(f"Predicted Emission Max: **{float(pred):.2f} nm**")


# =====================================================================================
# TAB 4 — FRET Analysis
# =====================================================================================
with tab4:
    st.header("🔬 FRET Pair Analysis")

    col1, col2 = st.columns(2)

    # donor input
    with col1:
        st.subheader("Donor")
        m = st.radio("Input:", ["SMILES", "JSME", "Upload"], key="d_in")
        donor_smi = ""
        if m == "SMILES":
            donor_smi = st.text_input("Donor SMILES:")
        elif m == "Upload":
            f = st.file_uploader("Upload Donor", type=["smi", "mol", "sdf"])
            if f:
                donor_smi = read_molecule_file(f)
        else:
            js = jsme_editor("donor")
            if js:
                donor_smi = urllib.parse.unquote(js)

    # acceptor input
    with col2:
        st.subheader("Acceptor")
        m = st.radio("Input:", ["SMILES", "JSME", "Upload"], key="a_in")
        acceptor_smi = ""
        if m == "SMILES":
            acceptor_smi = st.text_input("Acceptor SMILES:")
        elif m == "Upload":
            f = st.file_uploader("Upload Acceptor", type=["smi", "mol", "sdf"])
            if f:
                acceptor_smi = read_molecule_file(f)
        else:
            js = jsme_editor("acceptor")
            if js:
                acceptor_smi = urllib.parse.unquote(js)

    # Show structures
    if donor_smi:
        st.write("### Donor Structure")
        show_mol(donor_smi)

    if acceptor_smi:
        st.write("### Acceptor Structure")
        show_mol(acceptor_smi)

    # FRET logic
    if donor_smi or acceptor_smi:

        # Load dataset
        df = load_dataset()

        if df is None:
            st.error("Dataset not found.")
        else:
            st.write("### FRET Computation")

            solvent = "O"
            solv = smiles_to_maccs_df(solvent)

            if donor_smi:
                feat_d = smiles_to_maccs_df(donor_smi)
                X = pd.concat([feat_d, solv], axis=1)
                donor_em = predict(model_emission, X)
                donor_abs = predict(model_absorption, X)

                st.success(f"Donor Emission: {donor_em:.2f} nm")
                st.info(f"Donor Absorption: {donor_abs:.2f} nm")

                dfF = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "1", "true"])].copy()
                dfF["Δ"] = abs(dfF["AbsorptioMax (nm)"] - donor_em)
                top = dfF.sort_values("Δ").head(5)

                st.subheader("Top 5 FRET Acceptors")
                st.table(top[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ"]])

            if acceptor_smi:
                feat_a = smiles_to_maccs_df(acceptor_smi)
                X = pd.concat([feat_a, solv], axis=1)
                acceptor_abs = predict(model_absorption, X)

                st.success(f"Acceptor Absorption: {acceptor_abs:.2f} nm")

                dfF = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes", "1", "true"])].copy()
                dfF["Δ"] = abs(dfF["EmissionMax (nm)"] - acceptor_abs)
                top = dfF.sort_values("Δ").head(5)

                st.subheader("Top 5 FRET Donors")
                st.table(top[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ"]])


# Footer
st.write("---")
st.caption("FluroML © P. Deshmukh")
