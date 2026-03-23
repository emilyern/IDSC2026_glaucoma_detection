import streamlit as st
import numpy as np
from PIL import Image
import cv2
import os
import tf_keras as keras
from tf_keras.applications.efficientnet import preprocess_input

st.title("👁️ Glaucoma Detection")

@st.cache_resource
def load_model():
    return keras.models.load_model("fixed_model.keras")

model = load_model()

def preprocess_image(image):
    image = image.convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    return np.expand_dims(img_array, axis=0)

def make_gradcam_heatmap(img_array, model):
    import tensorflow as tf

    # Find EfficientNet base model
    base_model = None
    for layer in model.layers:
        if "efficientnet" in layer.name.lower():
            base_model = layer
            break

    if base_model is None:
        raise ValueError("EfficientNetB0 base model not found.")

    last_conv_layer = base_model.get_layer("top_conv")
    grad_model = keras.models.Model(
        inputs=base_model.inputs,
        outputs=[last_conv_layer.output, base_model.output]
    )

    img_tensor = tf.cast(img_array, tf.float32)

    with tf.GradientTape() as tape:
        conv_outputs, _ = grad_model(img_tensor, training=False)
        tape.watch(conv_outputs)

        x = conv_outputs
        for layer in model.layers[1:]:
            x = layer(conv_outputs if layer == model.layers[1] else x, training=False)

        loss = x[:, 0]

    grads = tape.gradient(loss, conv_outputs)

    if grads is None:
        raise ValueError("Gradients are None.")

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_outputs[0]
    heatmap = conv_out @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()

def overlay_gradcam(original_image_pil, heatmap, alpha=0.4):
    img = np.array(original_image_pil.convert("RGB").resize((224, 224)))
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET
    )
    superimposed = cv2.addWeighted(img_bgr, 1 - alpha, heatmap_colored, alpha, 0)
    superimposed_rgb = cv2.cvtColor(superimposed, cv2.COLOR_BGR2RGB)
    return Image.fromarray(superimposed_rgb)

uploaded_file = st.file_uploader(
    "Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    processed = preprocess_image(image)
    prediction = model.predict(processed)[0][0]

    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("Result")
        if prediction > 0.5:
            label = "Glaucoma (GON+)"
            st.error(f"**{label}**")
        else:
            label = "Normal (GON−)"
            st.success(f"**{label}**")

        confidence = prediction if prediction > 0.5 else 1 - prediction
        st.metric("Confidence", f"{confidence:.1%}")

    st.divider()
    if st.checkbox("Show GradCAM heatmap (which region influenced the prediction)"):
        try:
            heatmap = make_gradcam_heatmap(processed, model)
            gradcam_img = overlay_gradcam(image, heatmap)

            col3, col4 = st.columns(2)
            with col3:
                st.image(image.resize((224, 224)), caption="Original", use_container_width=True)
            with col4:
                st.image(gradcam_img, caption="GradCAM Heatmap", use_container_width=True)

            st.caption("🔴 Red = regions the model focused on | 🔵 Blue = low attention")
        except Exception as e:
            st.warning(f"GradCAM failed: {e}")

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