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
# ✔ 2D Molecule Viewer (MolView – cloud safe)
# =====================================================================================
def show_mol(smiles: str, width=450, height=350):
    """Display a 2D molecule using MolView iframe."""
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
# ✔ JSME Editor (full default toolbar)
# =====================================================================================
def jsme_editor(key: str, width=520, height=360, initial_smiles: str = "") -> str:
    param = f"jsme_{key}"
    params = st.experimental_get_query_params()

    if param in params:
        return params[param][0]

    jsme_js = "https://jsme-editor.github.io/dist/jsme.nocache.js"
    init = urllib.parse.quote(initial_smiles or "")

    html = f"""
    <div id="jsme_container_{key}" style="border:1px solid #ddd; width:{width}px; height:{height}px;"></div>
    <div style="margin-top:6px;">
      <button id="exp_{key}" style="margin-right:8px;">Export SMILES</button>
      <button id="clr_{key}">Clear</button>
    </div>

    <script src="{jsme_js}"></script>
    <script>
      function initJSME() {{
        try {{
          var jsme = new JSApplet.JSME("jsme_container_{key}", "{width-20}px", "{height-70}px");
          if ("{init}") {{
            try {{ jsme.setSmiles(decodeURIComponent("{init}")); }} catch(e){{}}
          }}

          document.getElementById("exp_{key}").onclick = function() {{
            var s = jsme.smiles();
            if(!s) {{ alert("Draw structure first"); return; }}
            var enc = encodeURIComponent(s);
            var u = new URL(window.location.href);
            u.searchParams.set("{param}", enc);
            window.location.href = u.toString();
          }};

          document.getElementById("clr_{key}").onclick = function() {{
            jsme.setSmiles("");
          }};
        }} catch(e) {{
          setTimeout(initJSME, 200);
        }}
      }}
      initJSME();
    </script>
    """

    components.html(html, height=height + 80)
    return ""


# =====================================================================================
# ✔ Streamlit Page Config
# =====================================================================================
st.set_page_config(
    page_title="FluroML: Molecular Fluorescence Predictor",
    layout="wide",
)


# =====================================================================================
# ✔ Load Models
# =====================================================================================
@st.cache_resource
def load_model(path):
    try:
        return joblib.load(path)
    except Exception as e:
        st.error(f"Failed to load {path}: {e}")
        return None

model_fluor = load_model("best_classifier_compatible.joblib")
model_abs   = load_model("new_best_regressor_compatible.joblib")
model_em    = load_model("best_regressor_emission_compatible.joblib")


# =====================================================================================
# ✔ Featurization
# =====================================================================================
def smiles_to_morgan(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    return dc.feat.CircularFingerprint(radius=3, size=1024).featurize([mol])[0]


def smiles_to_maccs(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        st.error("Invalid SMILES.")
        return None
    arr = dc.feat.MACCSKeysFingerprint().featurize([mol])
    return pd.DataFrame(arr)


def predict(model, feat):
    import numpy as np
    if model is None:
        return None
    X = feat.values if isinstance(feat, pd.DataFrame) else np.array(feat)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    try:
        out = model.predict(X)
        return out[0]
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return None


# =====================================================================================
# ✔ File loader
# =====================================================================================
def read_molecule_file(f):
    name = f.name.lower()
    data = f.getvalue().decode("utf-8", errors="ignore")

    if name.endswith(".smi"):
        return data.splitlines()[0].split()[0]

    if name.endswith(".mol") or name.endswith(".sdf"):
        mol = Chem.MolFromMolBlock(data)
        return Chem.MolToSmiles(mol) if mol else None

    st.error("Unsupported format.")
    return None


# =====================================================================================
# ✔ Dataset
# =====================================================================================
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("All Properties with Finguprints_3.csv")
    except Exception as e:
        st.error(e)
        return None


# =====================================================================================
# APP TABS
# =====================================================================================
st.title("FluroML: Molecular Fluorescence Predictor (JSME + 2D Structures)")

tab1, tab2, tab3, tab4 = st.tabs([
    "Fluorescence Classification",
    "Absorption Max Prediction",
    "Emission Max Prediction",
    "FRET Analysis",
])


# =====================================================================================
# TAB 1 — Fluorescence Classification
# =====================================================================================
with tab1:
    st.header("🧪 Fluorescence Classification")

    method = st.radio("Input Method:", 
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="fluoro_method_radio")

    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Enter SMILES:", key="fluoro_smi_input")

    elif method == "Upload File":
        f = st.file_uploader("Upload molecule file", type=["smi", "mol", "sdf"], key="fluoro_upload")
        if f:
            smi = read_molecule_file(f)

    else:
        js = jsme_editor("fluoro_js")
        if js:
            smi = urllib.parse.unquote(js)
            st.success("Molecule exported.")

    if smi:
        st.subheader("2D Structure")
        show_mol(smi)

        feat = smiles_to_morgan(smi)
        if feat is not None:
            pred = predict(model_fluor, feat)
            if pred is not None:
                st.success("Fluorescent" if int(pred) == 1 else "Non-Fluorescent")


# =====================================================================================
# TAB 2 — Absorption
# =====================================================================================
with tab2:
    st.header("🌈 Absorption Max Prediction")

    method = st.radio("Input Method:", 
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="abs_method_radio")

    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Enter SMILES:", key="abs_smi_input")

    elif method == "Upload File":
        f = st.file_uploader("Upload molecule file", type=["smi", "mol", "sdf"], key="abs_upload")
        if f:
            smi = read_molecule_file(f)

    else:
        js = jsme_editor("abs_js")
        if js:
            smi = urllib.parse.unquote(js)

    solvent = st.text_input("Solvent SMILES (e.g., O):", key="abs_solvent")

    if smi:
        st.subheader("2D Structure")
        show_mol(smi)

    if smi and solvent:
        m = smiles_to_maccs(smi)
        s = smiles_to_maccs(solvent)
        if m is not None and s is not None:
            X = pd.concat([m, s], axis=1)
            pred = predict(model_abs, X)
            if pred is not None:
                st.info(f"Predicted Absorption Max: **{float(pred):.2f} nm**")


# =====================================================================================
# TAB 3 — Emission
# =====================================================================================
with tab3:
    st.header("🔦 Emission Max Prediction")

    method = st.radio("Input Method:",
                      ["SMILES Input", "Draw Molecule (JSME)", "Upload File"],
                      key="em_method_radio")

    smi = ""

    if method == "SMILES Input":
        smi = st.text_input("Enter SMILES:", key="em_smi_input")

    elif method == "Upload File":
        f = st.file_uploader("Upload molecule file", type=["smi", "mol", "sdf"], key="em_upload")
        if f:
            smi = read_molecule_file(f)

    else:
        js = jsme_editor("em_js")
        if js:
            smi = urllib.parse.unquote(js)

    solvent = st.text_input("Solvent SMILES (e.g., O):", key="em_solvent")

    if smi:
        st.subheader("2D Structure")
        show_mol(smi)

    if smi and solvent:
        m = smiles_to_maccs(smi)
        s = smiles_to_maccs(solvent)
        if m is not None and s is not None:
            X = pd.concat([m, s], axis=1)
            pred = predict(model_em, X)
            if pred is not None:
                st.info(f"Predicted Emission Max: **{float(pred):.2f} nm**")


# =====================================================================================
# TAB 4 — FRET Analysis
# =====================================================================================
with tab4:
    st.header("🔬 FRET Pair Analysis")

    col1, col2 = st.columns(2)

    # ---------- Donor ----------
    with col1:
        st.subheader("Donor")
        m = st.radio("Donor Input:", 
                     ["SMILES", "JSME", "Upload"], 
                     key="donor_radio")

        donor = ""
        if m == "SMILES":
            donor = st.text_input("Donor SMILES:", key="donor_smi")
        elif m == "Upload":
            f = st.file_uploader("Upload Donor", type=["smi", "mol", "sdf"], key="donor_up")
            if f:
                donor = read_molecule_file(f)
        else:
            js = jsme_editor("donor_js")
            if js:
                donor = urllib.parse.unquote(js)

    # ---------- Acceptor ----------
    with col2:
        st.subheader("Acceptor")
        m = st.radio("Acceptor Input:", 
                     ["SMILES", "JSME", "Upload"],
                     key="acceptor_radio")

        acceptor = ""
        if m == "SMILES":
            acceptor = st.text_input("Acceptor SMILES:", key="acc_smi")
        elif m == "Upload":
            f = st.file_uploader("Upload Acceptor", type=["smi", "mol", "sdf"], key="acc_up")
            if f:
                acceptor = read_molecule_file(f)
        else:
            js = jsme_editor("acc_js")
            if js:
                acceptor = urllib.parse.unquote(js)

    # ---------- Show structures ----------
    if donor:
        st.write("### Donor Structure")
        show_mol(donor)

    if acceptor:
        st.write("### Acceptor Structure")
        show_mol(acceptor)

    # ---------- FRET computation ----------
    if donor or acceptor:
        df = load_dataset()
        if df is None:
            st.error("Dataset missing.")
        else:
            solvent = "O"
            s = smiles_to_maccs(solvent)

            # donor mode
            if donor:
                d = smiles_to_maccs(donor)
                if d is not None:
                    X = pd.concat([d, s], axis=1)
                    donor_em = predict(model_em, X)
                    donor_abs = predict(model_abs, X)

                    st.success(f"Donor Emission: {donor_em:.2f} nm")
                    st.info(f"Donor Absorption: {donor_abs:.2f} nm")

                    df2 = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes","true","1"])].copy()
                    df2["Δ"] = abs(df2["AbsorptioMax (nm)"] - donor_em)

                    st.subheader("Top 5 FRET Acceptors")
                    st.table(df2.sort_values("Δ").head(5)[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ"]])

            # acceptor mode
            if acceptor:
                a = smiles_to_maccs(acceptor)
                if a is not None:
                    X = pd.concat([a, s], axis=1)
                    acc_abs = predict(model_abs, X)

                    st.success(f"Acceptor Absorption: {acc_abs:.2f} nm")

                    df2 = df[df["Fluorescent labeling"].astype(str).str.lower().isin(["yes","true","1"])].copy()
                    df2["Δ"] = abs(df2["EmissionMax (nm)"] - acc_abs)

                    st.subheader("Top 5 FRET Donors")
                    st.table(df2.sort_values("Δ").head(5)[["Smiles", "AbsorptioMax (nm)", "EmissionMax (nm)", "Δ"]])

# Footer
st.write("---")
st.caption("FluroML © P. Deshmukh")
