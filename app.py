import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

st.title("👁️ Glaucoma Detection")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("fixed_model.keras")

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)  
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed = preprocess_image(image)  
    prediction = model.predict(processed)[0][0]

    if prediction > 0.5:
        label = "Glaucoma"
        st.error(f"**Prediction: {label}**")
    else:
        label = "Normal"
        st.success(f"**Prediction: {label}**")

st.info("⚠️ For educational purposes only. Not a clinical diagnostic tool.")

# ----------------------------
# Dataset Citation Section
st.divider()
st.subheader("📚 Dataset Citation & Info")
st.info("This AI model was trained using the Hillel Yaffe Glaucoma Dataset (PhysioNet).")

citation = """@misc{yaffe2024glaucoma,
  title={Hillel Yaffe Glaucoma Dataset},
  author={Abramovich, Hadas Pizem, Jonathan Fhima, Eran Berkowitz, Ben Gofrit, Jan Van Eijgen, Eytan Blumenthal, Joachim Behar },
  year={2024},
  howpublished={\\url{https://physionet.org/content/hillel-yaffe-glaucoma-dataset/1.0.0/}},
  note={This dataset was used to train the model in this project}
}"""
st.code(citation, language="bibtex")