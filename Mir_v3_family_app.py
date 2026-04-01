import streamlit as st
import pandas as pd
import joblib
import re

st.set_page_config(page_title="Mir v3 — Family Model", page_icon="🧬")
st.title("🧬 miRNA Upregulation Predictor — v3 Family")
st.caption("Random Forest · OneHotEncoder · Seed family · No scenario feature")

@st.cache_resource
def load_model():
    return joblib.load('Mir_v3_family_model.pkl')

bundle = load_model()
model_pipeline   = bundle['model']
mirna_lookup     = bundle['mirna_lookup']
accession_lookup = bundle['accession_lookup']
options          = bundle['options']

if 'history' not in st.session_state:
    st.session_state.history = []

def normalize(name: str) -> str:
    name = name.strip()
    return re.sub(r'-(5p|3p)$', '', name.lower())

def resolve_mirna(user_input: str):
    user_input = user_input.strip()
    if user_input in accession_lookup:
        e = accession_lookup[user_input]
        return e['microrna_group_simplified'], e['family_name'], user_input
    if user_input in mirna_lookup:
        e = mirna_lookup[user_input]
        return e['microrna_group_simplified'], e['family_name'], e.get('mirbase_accession')
    norm_input = normalize(user_input)
    for key, val in mirna_lookup.items():
        if normalize(key) == norm_input:
            return val['microrna_group_simplified'], val['family_name'], val.get('mirbase_accession')
    return None

# ── Inputs
# ── Inputs
st.subheader("Enter Prediction Inputs")
mirna_input = st.text_input("miRNA name or accession number",
                             placeholder="e.g. hsa-miR-21, miR-155, MIMAT0000076")
parasite    = st.selectbox("Parasite",  options['parasite'])
organism    = st.selectbox("Organism",  options['organism'])
cell_type   = st.selectbox("Cell type", options['cell_type'])

time = st.number_input(
    "Time (hours post-infection)",
    min_value=int(min(options['time'])),
    value=int(options['time'][0]),
    step=1
)
resolved = None
if mirna_input:
    resolved = resolve_mirna(mirna_input)
    if resolved:
        group, family, accession = resolved
        fam_display = family if family != 'unknown_family' else 'Not conserved'
        st.success(f"**miRNA group:** `{group}`")
        col1, col2 = st.columns(2)
        col1.metric("Family name", fam_display)
        col2.metric("Accession",   accession or "N/A")
    else:
        st.warning("miRNA not found in lookup. Group will be derived from name.")

if st.button("Predict", type="primary"):
    if not mirna_input.strip():
        st.warning("Please enter a miRNA name.")
    else:
        if resolved:
            group, family, _ = resolved
        else:
            group  = re.sub(r'^[a-z]{3}-', '', mirna_input.strip().lower())
            group  = re.sub(r'-(5p|3p)$', '', group)
            family = 'unknown_family'

        fam_val = None if family == 'unknown_family' else family

        input_df = pd.DataFrame([{
            'microrna_group_simplified': group,
            'parasite':                 parasite,
            'organism':                 organism,
            'cell type':                cell_type,
            'family_name':              fam_val,
            'time':                     int(time),
            'is_conserved':             0 if fam_val is None else 1,
        }])

        proba = model_pipeline.predict_proba(input_df)[0][1]
        pred  = int(proba >= 0.5)
        label = "⬆️ Upregulated" if pred == 1 else "⬇️ Downregulated"
        color = "green" if pred == 1 else "red"

        st.markdown(f"### Prediction: :{color}[{label}]")
        st.metric("Confidence", f"{proba*100:.1f}%")

        fam_display = family if family != 'unknown_family' else 'Not conserved'
        st.session_state.history.append({
            "miRNA":      mirna_input.strip(),
            "Group":      group,
            "Family":     fam_display,
            "Parasite":   parasite,
            "Organism":   organism,
            "Cell type":  cell_type,
            "Time (h)":   time,
            "Prediction": label,
            "Confidence": f"{proba*100:.1f}%",
        })

if st.session_state.history:
    st.subheader("Prediction History")
    st.dataframe(pd.DataFrame(st.session_state.history), use_container_width=True)
    if st.button("Clear history"):
        st.session_state.history = []
