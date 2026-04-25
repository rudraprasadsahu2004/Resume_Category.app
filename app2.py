# =========================
# 🔹 INSTALL REQUIREMENTS
# =========================
# pip install streamlit scikit-learn python-docx PyPDF2

import streamlit as st
import pickle
import docx
import PyPDF2
import re
import base64

# =========================
# 🔹 LOAD MODEL FILES
# =========================
svc_model = pickle.load(open('clf.pkl', 'rb'))
tfidf = pickle.load(open('tfidf.pkl', 'rb'))
le = pickle.load(open('encoder.pkl', 'rb'))

# =========================
# 🔹 LOAD LOGO IMAGE
# =========================
def get_base64_of_image(image_file):
    with open(image_file, "rb") as f:
        return base64.b64encode(f.read()).decode()

img_base64 = get_base64_of_image("images.jpg")

# =========================
# 🔹 TEXT CLEANING
# =========================
def cleanResume(txt):
    txt = re.sub('http\S+\s', ' ', txt)
    txt = re.sub('RT|cc', ' ', txt)
    txt = re.sub('#\S+\s', ' ', txt)
    txt = re.sub('@\S+', ' ', txt)
    txt = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', txt)
    txt = re.sub(r'[^\x00-\x7f]', ' ', txt)
    txt = re.sub('\s+', ' ', txt)
    return txt

# =========================
# 🔹 FILE READ FUNCTIONS
# =========================
def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return "".join([page.extract_text() or "" for page in reader.pages])

def extract_text_from_docx(file):
    doc = docx.Document(file)
    return "\n".join([p.text for p in doc.paragraphs])

def extract_text_from_txt(file):
    try:
        return file.read().decode('utf-8')
    except:
        return file.read().decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type")

# =========================
# 🔹 PREDICTION
# =========================
def pred(text):
    cleaned = cleanResume(text)
    vec = tfidf.transform([cleaned]).toarray()
    result = svc_model.predict(vec)
    return le.inverse_transform(result)[0]

# =========================
# 🔹 MAIN UI
# =========================
def main():
    st.set_page_config(page_title="Resume AI", page_icon="📄", layout="wide")

    # =========================
    # 🔹 RESPONSIVE CSS
    # =========================
    st.markdown(f"""
    <style>
    /* Mobile responsive text */
    @media (max-width: 768px) {{
        .title-text {{
            font-size: 20px !important;
        }}
        .main-title {{
            font-size: 22px !important;
        }}
    }}

    .header-container {{
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 10px;
    }}

    .logo-img {{
        width: 45px;
        height: 45px;
        border-radius: 50%;
    }}

    .title-text {{
        font-size: 26px;
        font-weight: bold;
    }}

    .main-title {{
        font-size: 28px;
        font-weight: 600;
        text-align: center;
    }}

    .center {{
        text-align: center;
    }}
    </style>

    <div class="header-container">
        <img src="data:image/png;base64,{img_base64}" class="logo-img">
        <div class="title-text">Rudra's AI</div>
    </div>
    """, unsafe_allow_html=True)

    # =========================
    # 🔹 TITLE
    # =========================
    st.markdown('<div class="main-title">📄 Resume Category Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="center">Upload your resume and get AI prediction instantly</div>', unsafe_allow_html=True)

    st.write("")

    # =========================
    # 🔹 FILE UPLOAD
    # =========================
    uploaded_file = st.file_uploader("Upload Resume", type=["pdf", "docx", "txt"])

    if uploaded_file:
        try:
            resume_text = handle_file_upload(uploaded_file)
            st.success("✅ Resume uploaded successfully!")

            # Toggle text view
            with st.expander("📄 View Extracted Text"):
                st.text_area("", resume_text, height=250)

            # Predict button (better for mobile)
            if st.button("🔍 Predict Category"):
                category = pred(resume_text)

                st.markdown("### 🎯 Prediction Result")
                st.success(f"**{category}**")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# =========================
# 🔹 RUN APP
# =========================
if __name__ == "__main__":
    main()