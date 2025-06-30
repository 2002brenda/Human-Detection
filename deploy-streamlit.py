import streamlit as st
import cv2
import tempfile
import os
import time
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load YOLOv8 model
model_path = r"runs\detect\train\weights\best.pt"  # Ganti dengan path model Anda
model = YOLO(model_path)

# (Opsional) Class model — Hapus tampilan ke UI
CLASS_NAMES = model.names

# Streamlit page configuration with pastel colors and custom background
st.set_page_config(page_title="Human Detection App", layout="centered")

# Custom CSS for colors based on your image
st.markdown(
    """
    <style>
    /* Latar belakang hitam utama */
    div[data-testid="stApp"] {
        background-color: #000000; /* Black */
        color: #FAF1E0 !important; /* White cream for text */
    }

    /* Warna tombol */
    button[kind="primary"] {
        background-color: #800000 !important;  /* Dark Red */
        color: #FAF1E0 !important;
        border: none;
        border-radius: 10px;
    }

    button[kind="secondary"] {
        background-color: #F4E1C1 !important;  /* Pastel Cream */
        color: #000000 !important;
        border: none;
        border-radius: 10px;
    }

    /* Tombol hover effect */
    button:hover {
        background-color: #6B0000 !important; /* Darker Red for hover effect */
        color: #FAF1E0 !important;
    }

    /* Ubah warna teks judul dan label input */
    h1, h2, h3, h4, h5, h6 {
        color: #FAF1E0 !important; /* White cream for titles */
    }

    label, .css-1kyxreq, .css-1n76uvr {
        color: #000000 !important; /* Semua teks input menjadi hitam */
    }

    /* Dropdown dan input box */
    .stNumberInput, .stTextInput, .stSelectbox {
        background-color: #F4E1C1 !important; /* Pastel Cream for input boxes */
        color: #000000 !important;
        border-radius: 8px !important;
    }

    /* Menyembunyikan sidebar jika diperlukan */
    .css-1gw4f64 {
        background-color: #000000 !important;
        color: #FAF1E0 !important;
    }

    .css-1kyxreq {
        color: #FAF1E0 !important;
    }

    /* Ubah warna "People" menjadi merah */
    .css-1g1v14v {
        color: #FF0000 !important;  /* Red color for "People:" text */
    }
    </style>
    """, 
    unsafe_allow_html=True
)

# Title and description
st.title("Human Detection using YOLOv8")
st.markdown("Detect people in images, videos, and webcam feed.")

# Option to choose input type
option = st.radio("Select input type:", ["Image", "Video", "Webcam"])

# Input for setting maximum number of people to detect
max_people = st.number_input("Set maximum number of people to detect:", min_value=1, max_value=100, value=10)

# Function to draw results and count people
def draw_results(frame, results):
    person_count = 0  # Initialize person count for each frame
    for r in results:
        boxes = r.boxes  # Get the bounding boxes for detected objects
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Coordinates of the bounding box
            conf = box.conf[0]  # Confidence level of the detection
            cls = int(box.cls[0])  # Class ID of the detected object
            label_name = model.names[cls].lower()
            
            if label_name in ["person", "people"]:
                person_count += 1
                label = f"{label_name.capitalize()} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

    # Display the total number of detected people on the frame in red color
    cv2.putText(
        frame,
        f"People: {person_count}",  # Display total people detected in the frame
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 0, 255),  # Red color for the count
        2,
    )

    return frame, person_count  # Return the updated frame and the count of detected people

# Handling Image Input
if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        # Run detection
        results = model.predict(image, imgsz=640)
        result_image, person_count = draw_results(np.array(image.copy()), results)

        st.image(result_image, caption="Detection Result", use_container_width=True)
        st.write(f"People Detected: {person_count}")

# Handling Video Input
elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])
    if uploaded_video:
        # Create a temporary directory to store our files
        temp_dir = tempfile.mkdtemp()
        input_path = os.path.join(temp_dir, "input_video.mp4")
        output_path = os.path.join(temp_dir, "output_video.mp4")

        # Save the uploaded video to the temporary file
        with open(input_path, "wb") as f:
            f.write(uploaded_video.getbuffer())

        # Show original video
        st.video(uploaded_video)
        st.write("Original Video ⬆")

        # Button to start processing
        if st.button("Start Detection on Video"):
            # Process the video
            cap = cv2.VideoCapture(input_path)
            if not cap.isOpened():
                st.error("Error: Could not open video file.")
            else:
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # Always use mp4v codec for MP4 output
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

                # Add progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()

                # Process frames
                frame_count = 0
                preview_frame = None

                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Update progress
                    frame_count += 1
                    progress = int(frame_count / total_frames * 100)
                    progress_bar.progress(progress)
                    status_text.text(
                        f"Processing frame {frame_count}/{total_frames} ({progress}%)"
                    )

                    # Process frame
                    results = model.predict(frame, imgsz=640)
                    annotated_frame, person_count = draw_results(frame.copy(), results)
                    out.write(annotated_frame)

                    # Store last frame for preview
                    preview_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Release resources
                cap.release()
                out.release()

                try:
                    # Check if we need to convert the video format
                    final_output_path = os.path.join(temp_dir, "final_output.mp4")

                    try:
                        # Try to use FFmpeg for conversion to ensure browser compatibility
                        import subprocess

                        ffmpeg_cmd = [
                            "ffmpeg",
                            "-y",
                            "-i",
                            output_path,
                            "-vcodec",
                            "libx264",
                            "-pix_fmt",
                            "yuv420p",  # Ensure compatibility
                            "-preset",
                            "fast",
                            final_output_path,
                        ]
                        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                        video_path_to_use = final_output_path
                    except (ImportError, subprocess.SubprocessError, FileNotFoundError):
                        # If FFmpeg fails or isn't available, use the original output
                        st.warning(
                            "FFmpeg conversion not available. Video playback might not work in all browsers."
                        )
                        video_path_to_use = output_path

                    # Read the video as bytes
                    with open(video_path_to_use, "rb") as file:
                        video_bytes = file.read()

                    st.success(
                        "✅ Detection complete. Video ready to view and download."
                    )

                    st.write(f"Processed Video (People Detected: {person_count}):")
                    st.video(video_bytes)

                    st.download_button(
                        label="Download Processed Video (MP4)",
                        data=video_bytes,
                        file_name="processed_video.mp4",
                        mime="video/mp4",
                    )
                except Exception as e:
                    st.error(f"❌ Failed to process or display video: {e}")
                    st.error("Try downloading the video and playing it locally.")

# Handling Webcam Input
elif option == "Webcam":
    st.markdown("### Webcam Detection")
    st.warning("Tekan START untuk memulai deteksi dan STOP untuk mengakhiri")

    # Initialize session state
    if 'webcam_state' not in st.session_state:
        st.session_state.webcam_state = {
            'active': False,
            'cap': None,
            'total': 0,
            'frame_count': 0
        }

    # Create a single button that toggles between start/stop
    if not st.session_state.webcam_state['active']:
        if st.button("START WEBCAM", key='start_btn', type='primary'):
            st.session_state.webcam_state['active'] = True
            st.session_state.webcam_state['cap'] = cv2.VideoCapture(0)
            st.session_state.webcam_state['total'] = 0
            st.session_state.webcam_state['frame_count'] = 0
            st.rerun()
    else:
        if st.button("STOP WEBCAM", key='stop_btn', type='secondary'):
            if st.session_state.webcam_state['cap']:
                st.session_state.webcam_state['cap'].release()
            st.session_state.webcam_state['active'] = False
            st.success(f"Deteksi selesai! Total orang terdeteksi: {st.session_state.webcam_state['total']}")
            st.rerun()

    # Webcam processing
    if st.session_state.webcam_state['active']:
        frame_placeholder = st.empty()
        status_text = st.empty()
        cap = st.session_state.webcam_state['cap']

        if cap.isOpened():
            while st.session_state.webcam_state['active']:
                ret, frame = cap.read()
                if not ret:
                    status_text.error("Gagal mengambil frame dari webcam")
                    break

                # Process detection
                results = model.predict(frame, imgsz=640, conf=0.4)
                processed_frame, count = draw_results(frame.copy(), results)

                # Update counters in session state
                st.session_state.webcam_state['total'] += count
                st.session_state.webcam_state['frame_count'] += 1

                # Display the frame with annotations
                frame_placeholder.image(
                    cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB),
                    channels="RGB",
                )

                # Warning if the total detected people exceed the set maximum
                if count > max_people:
                    st.warning(f"Warning: More than {max_people} people detected in frame! Actual count: {count}")

                time.sleep(0.01)  # Reduce CPU usage
        else:
            st.error("Tidak dapat mengakses webcam")
            st.session_state.webcam_state['active'] = False
