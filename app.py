import streamlit as st
import requests
from PIL import Image, ImageDraw

st.set_page_config(
    page_title="Multi-View Grid AI Detector", page_icon="🛡️", layout="centered"
)

st.title("🛡️ Spatial Multi-View AI Detector")
st.write("Upload an image canvas to generate a localized forensic heat-map analysis.")

uploaded_file = st.file_uploader(
    "Choose an image canvas...", type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Open image locally to draw masks directly onto it
    base_image = Image.open(uploaded_file).convert("RGBA")
    st.image(uploaded_file, caption="Original Uploaded Asset", use_container_width=True)

    if st.button("Generate Patch Map Matrix"):
        with st.spinner(
            "Executing structural slice evaluations across framework grid..."
        ):
            try:
                files = {
                    "file": (
                        uploaded_file.name,
                        uploaded_file.getvalue(),
                        uploaded_file.type,
                    )
                }
                response = requests.post("http://127.0.0.1:8000/api_post/", files=files)

                if response.status_code == 200:
                    results_data = response.json().get("patch_evaluations", [])

                    # Create an overlay layer for alpha-blended transparency
                    overlay = Image.new("RGBA", base_image.size, (0, 0, 0, 0))
                    draw = ImageDraw.Draw(overlay)

                    fake_count = 0
                    real_count = 0

                    for p in results_data:
                        left, top, right, bottom = p["box"]
                        pred = p["prediction"]
                        conf = p["confidence"]

                        if "Fake" in pred or "Synthetic" in pred:
                            # Semi-transparent red for fake segments (alpha = 100 out of 255)
                            draw.rectangle(
                                [left, top, right, bottom],
                                fill=(239, 68, 68, 100),
                                outline=(239, 68, 68, 255),
                                width=2,
                            )
                            fake_count += 1
                        else:
                            # Semi-transparent green for real segments (alpha = 60 out of 255)
                            draw.rectangle(
                                [left, top, right, bottom],
                                fill=(34, 197, 94, 60),
                                outline=(34, 197, 94, 255),
                                width=2,
                            )
                            real_count += 1

                    # Combine original image with the semi-transparent overlay
                    final_visual = Image.alpha_composite(base_image, overlay)

                    st.markdown("---")
                    st.subheader("Localized Forensic Heatmap Overlay")
                    st.image(
                        final_visual,
                        caption="Forensic Localization Mask Maping",
                        use_container_width=True,
                    )

                    # Display grid analytics summary metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric(
                            label="AI Corrupted Patches Detected", value=fake_count
                        )
                    with col2:
                        st.metric(
                            label="Authentic Structural Patches", value=real_count
                        )

                    if fake_count > 0:
                        st.error(
                            f"⚠️ Warning: Structural anomalies detected in {fake_count} localized matrix regions."
                        )
                    else:
                        st.success(
                            "✅ Complete Asset Scan: No localized anomaly arrays observed."
                        )

                else:
                    st.error(f"Backend Error: {response.status_code}")
            except Exception as e:
                st.error("Frontend visualization compilation failed.")
                st.exception(e)
