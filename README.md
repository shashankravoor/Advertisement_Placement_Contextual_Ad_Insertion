# Advertisement Placement: Contextual Ad Insertion

This project is a Streamlit application that intelligently places advertisements into videos based on the video's content. It analyzes video and audio to find the most relevant and opportune moments to insert ads, creating a seamless and effective viewing experience.

## 🚀 Features

*   **Deep Content Analysis:** Leverages state-of-the-art AI models to understand video content:
    *   **Object Detection:** Identifies objects in video frames using YOLOv8.
    *   **Scene Understanding:** Determines the context of scenes using Qwen-VL.
    *   **Audio Transcription:** Transcribes spoken words using OpenAI's Whisper.
*   **Intelligent Ad Matching:**
    *   Analyzes ad content to create a comprehensive ad database.
    *   Uses a weighted scoring system to match ads to the most relevant video segments based on category, objects, and audio context.
*   **Interactive UI:** A user-friendly Streamlit interface to:
    *   Upload and preview videos.
    *   Configure analysis options.
    *   Review and select recommended ads.
    *   Generate the final video with ads seamlessly integrated.

## Application overview
You can check the application working by navigating to the advertisement_placement.mp4 file.


## 🛠️ Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/shashankravoor/Advertisement_Placement_Contextual_Ad_Insertion.git
    cd Advertisement_Placement_Contextual_Ad_Insertion
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```
    Install the required packages. You can check the main packages to be installed which is mentioned below

## 📦 Dependencies

This project relies on several key libraries:

*   `streamlit`
*   `torch`
*   `torchvision`
*   `torchaudio`
*   `transformers`
*   `ultralytics`
*   `openai-whisper`
*   `sentence-transformers`
*   `opencv-python-headless`
*   `pyscenedetect`
*   `ffmpeg-python`
*   `moviepy`
*   `pandas`
*   `numpy`
*   `plotly`
*   `seaborn`

You will also need to have `ffmpeg` installed on your system.

## ▶️ How to Run

1.  **Run the Streamlit application:**
    ```bash
    streamlit run smart_placement.py
    ```

2.  **Open your browser:**
    Navigate to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  **Using the App:**
    *   Select a video from the dropdown menu.
    *   Choose the desired analysis models (Object Detection, Scene Analysis, Audio Analysis).
    *   Define the time slots for potential ad placements.
    *   Click "Start Processing" to begin the analysis.
    *   Once the analysis is complete, the application will recommend the best-matching ads for each slot.
    *   Review the recommendations, make your selection, and generate the final video.

## 📁 Project Structure

```
.
├── smart_placement.py         # Main Streamlit application
├── assets/
│   ├── sample_videos/         # Source videos for ad insertion
│   ├── ads/                   # Directory for advertisement videos
│   │   └── Brands/            # Ads categorized by brand
│   └── ad_analyses/           # Cached analysis data for ads
├── outputs/
│   └── processed_videos/      # Output directory for final videos
├── yolov8n.pt                 # YOLOv8 model weights
└── README.md                  # Description of the project.
```
note:- To download the assets folder, visit this google drive [link](https://drive.google.com/file/d/1NLYcJELq6Jm5ESKuoyOQGPPR_KqY5t4J/view?usp=sharing) to download. 
