import streamlit as st
import pandas as pd
import tempfile
from license_plate_backend import process_video

st.set_page_config(
    page_title="AI Plate Recognition",
    layout="wide"
)

st.title("AI License Plate Detection Dashboard")

st.markdown(
"""
Upload a vehicle video to detect **license plates using YOLO + EasyOCR**
"""
)

uploaded_video = st.file_uploader(
    "Upload Video",
    type=["mp4","mov","avi"]
)

if uploaded_video:

    st.video(uploaded_video)

    temp = tempfile.NamedTemporaryFile(delete=False)
    temp.write(uploaded_video.read())

    progress_bar = st.progress(0)
    status = st.empty()

    if st.button("Start AI Detection"):

        def update_progress(value):
            progress_bar.progress(value)
            status.text(f"Processing: {int(value*100)}%")

        results = process_video(
            temp.name,
            progress_callback=update_progress
        )

        st.success("Detection Completed")

        if results:

            df = pd.DataFrame(results)

            df_unique = df.drop_duplicates(subset="plate")

            st.subheader("Detection Table")

            st.dataframe(
                df_unique[["plate","timestamp"]],
                use_container_width=True
            )

            st.subheader("Plate Image Gallery")

            cols = st.columns(4)

            for i,row in df_unique.iterrows():

                with cols[i%4]:

                    st.image(
                        row["image"],
                        caption=row["plate"],
                        width=250
                    )

            st.subheader("Plate Visible Time Analytics")

            chart = df["plate"].value_counts()

            st.bar_chart(chart)

            csv = df_unique.to_csv(index=False).encode("utf-8")

            st.download_button(
                "Download Detection CSV",
                csv,
                "plates.csv",
                "text/csv"
            )

        else:

            st.warning("No plates detected.")