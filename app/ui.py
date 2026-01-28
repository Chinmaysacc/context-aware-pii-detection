import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/analyze"

st.set_page_config(page_title="Context-Aware PII Detection", layout="wide")

st.title("🔐 Context-Aware PII Detection")
st.write("NER → Context Understanding → Human-readable Reasoning")

text_input = st.text_area(
    "Enter text to analyze",
    height=150,
    placeholder="Example: My email is john@example.com"
)

analyze_btn = st.button("Analyze")

if analyze_btn and text_input.strip():
    with st.spinner("Analyzing..."):
        response = requests.post(API_URL, json={"text": text_input})

    if response.status_code != 200:
        st.error("API error. Make sure FastAPI server is running.")
    else:
        data = response.json()

        st.subheader("📌 Context Analysis")
        st.write(f"**Context:** `{data['results'][0]['context_label']}`")
        st.write(f"**Confidence:** `{data['results'][0]['context_confidence']:.2f}`")

        st.subheader("🔍 Detected Entities")

        if data["entities_found"] == 0:
            st.success("No PII detected.")
        else:
            for entity in data["results"]:
                if entity["is_pii"]:
                    st.error(
                        f"🔴 **{entity['entity_type']}** → `{entity['entity_value']}`"
                    )
                else:
                    st.success(
                        f"🟢 **{entity['entity_type']}** → `{entity['entity_value']}`"
                    )

                with st.expander("Why?"):
                    st.write(entity["reasoning"])

else:
    st.info("Enter text and click **Analyze**")
