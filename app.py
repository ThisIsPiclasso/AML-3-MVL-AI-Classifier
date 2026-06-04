import streamlit as st
import requests

st.set_page_config(
    page_title="Multi-View AI Detector", page_icon="🛡️", layout="centered"
)

st.title("🛡️ Multi-View AI Image Detector")
st.write(
    "Upload an image below to evaluate it for synthetic or AI-generated manipulation signatures."
)

# File uploader widget
uploaded_file = st.file_uploader(
    "Choose an image canvas...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Display the uploaded image nicely on the screen
    st.image(uploaded_file, caption="Uploaded Image Asset", use_container_width=True)

    if st.button("Analyze Image Framework"):
        with st.spinner(
            "Extracting forensic noise residuals and frequency domain profiles..."
        ):
            try:
                # Prepare payload
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }

                # Query the backend
                response = requests.post("http://127.0.0.1:8000/api_post/", files=files)

                if response.status_code == 200:
                    result = response.json()

                    # Extract variables safely using data fallbacks
                    prediction = result.get("prediction", "Unknown")
                    confidence = float(result.get("confidence", 0.0))

                    st.markdown("---")
                    st.subheader("Forensic Evaluation Results")

                    # Render distinct visual alert blocks based on classification
                    if "Fake" in prediction or "Synthetic" in prediction:
                        st.error(f"🚨 **Verdict: {prediction}**")
                    else:
                        st.success(f"✅ **Verdict: {prediction}**")

                    # Clean metrics layout
                    st.metric(
                        label="Model Certainty Score", value=f"{confidence * 100:.2f}%"
                    )

                else:
                    st.error(
                        f"Backend Server Error: Status Code {response.status_code}"
                    )
                    st.info(f"Details: {response.text}")

            except Exception as e:
                st.error("Frontend parsing error encountered.")
                st.exception(
                    e
                )  # This will print the exact traceback cleanly right on the webpage if it fails again
