import streamlit as st
from transformers import pipeline

# 🔧 Must be first!
st.set_page_config(page_title="🛍️ AI Shopping Assistant", layout="centered")

# 🚀 Load smarter model
@st.cache_resource
def load_model():
    return pipeline(
        "text2text-generation",
        model="MBZUAI/LaMini-Flan-T5-783M"
    )

generator = load_model()

st.title("🛍️ Smart AI Shopping Assistant")

user_input = st.text_input("Ask anything about shopping:", placeholder="e.g., Best phones under ₹15000")

if user_input:
    with st.spinner("Finding the best deals..."):
        try:
            prompt = f"Give 3 detailed shopping suggestions for: {user_input}"
            output = generator(prompt, max_length=256, do_sample=True, temperature=0.7)[0]['generated_text']
            st.success(output)
        except Exception as e:
            st.error(f"❌ Error: {e}")
