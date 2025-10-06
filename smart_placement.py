import streamlit as st
import os
import seaborn as sns
import plotly.graph_objects as go
import random
import tempfile
import re
import cv2
import numpy as np
import pandas as pd
import plotly.express as px
from PIL import Image
import time
import json
import torch
from sentence_transformers import util
from datetime import datetime
import torch.backends.cudnn as cudnn
from transformers import AutoModelForCausalLM, AutoTokenizer
import whisper
from ultralytics import YOLO
from sentence_transformers import SentenceTransformer
import warnings
import logging
from transformers import TextStreamer
from typing import Union, List, Dict, Any,Set
import subprocess
from typing import Tuple
import matplotlib.pyplot as plt
from sentence_transformers import util
import plotly.express as px
import matplotlib.pyplot as plt
import joblib
import ffmpeg
import shutil
from collections import deque
import time
import threading
import psutil
import GPUtil
from pydantic import BaseModel, Field
from typing import List, Dict
from glob import glob
from pathlib import Path
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector 
import pandas as pd
import hashlib
import numpy as np
from sklearn.preprocessing import normalize
from moviepy import VideoFileClip, concatenate_videoclips
import streamlit.components.v1 as components
from collections import Counter


# Supporting constants (should be defined at module level)
WEIGHT_PRESETS = {
    "broadcast": {
        "category": 0.4,
        "objects": 0.3,
        "audio": 0.25,
        "ctr": 0.05
    },
    "product_review": {
        "category": 0.5,
        "objects": 0.4,
        "audio": 0.05,
        "ctr": 0.05
    },
    "generic": {
        "category":0.4,
        "objects":0.4,
        "audio":0.1,
        "ctr":0.05,
        "duration":0.05
    },
    "visual_heavy": {
        "category": 0.3,
        "objects": 0.5,
        "audio": 0.1,
        "ctr": 0.1
    }
}

# Add these near the top of your script, after imports but before class definitions
MODULES = ["Ad Management", "Video Segmentation", "Advertisement Analytics", "Content Matching"]
VIDEO_FOLDER = "assets/sample_videos"  # Define your video folder path
ADS_VIDEO_FOLDER = "assets/ads" # "assets/ads"  # Define your ads folder path
ASSETS_folder = "assets"  # Define your assets folder path
# ADS_VIDEO_FOLDER_JSON = "assets/ads_json"  # For JSON files

# Add the AD_FORMATS constant with your other constants
AD_FORMATS = {
    'pre-roll': {
        'max_duration': 120, 
        'interruption': 'high', 
        'position_control': 'fixed', 
        'content_matching': False
    },
    'mid-roll': {
        'max_duration': 120, 
        'interruption': 'medium', 
        'position_control': 'time-based', 
        'content_matching': True
    },
    'post-roll': {
        'max_duration': 120, 
        'interruption': 'low', 
        'position_control': 'fixed', 
        'content_matching': False
    },
    'overlay': {
        'max_duration': 120, 
        'interruption': 'minimal', 
        'position_control': 'pixel_coordinates', 
        'content_matching': True
    },
    'sponsored-segment': {
        'max_duration': 120, 
        'interruption': 'variable', 
        'position_control': 'time-based', 
        'content_matching': True
    }
}


# Then continue with your other constants
SUPPORTED_VIDEO_FORMATS = ['mp4', 'mov', 'avi', 'mkv']
MODEL_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Initialize SBERT model (you can put this in your init_session_state() or load_models() function)
sbert_model = SentenceTransformer('all-MiniLM-L6-v2').to(MODEL_DEVICE)

FRAME_SAVE_DIR = os.path.join("outputs", "processed_videos", "frames")
os.makedirs(FRAME_SAVE_DIR, exist_ok=True)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('video_processing.log')
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")

# Memory optimization settings
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
torch.cuda.empty_cache()



def check_environment():
    """Check and log environment configuration"""
    logger.info("===== Environment Check =====")
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA memory: Allocated: {torch.cuda.memory_allocated()/1024**2:.2f}MB, "
                   f"Cached: {torch.cuda.memory_reserved()/1024**2:.2f}MB")
    logger.info(f"OpenCV version: {cv2.__version__}")
    logger.info(f"NumPy version: {np.__version__}")
    logger.info("============================")


def init_session_state():
    session_defaults = {
        'current_project': None,
        'models_loaded': False,
        'models': None,
        'video_path': None,
        'video_name': None,
        'video_analysis': None,
        'ads': None,
        'selected_ads': [],
        'vector_recommendations': None,
        'current_video_id': None,
        'recommender': None
    }
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if 'embedding_stats' not in st.session_state:
        st.session_state.embedding_stats = None

def detect_scene_changes(video_path: str, threshold: float = 30.0) -> List[float]:
    """
    Detect scene changes using frame differences with histogram comparison.
    
    Args:
        video_path: Path to the video file
        threshold: Difference threshold for detecting scene changes (higher = fewer detections)
    
    Returns:
        List of timestamps (in seconds) where scene changes occur
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    prev_hist = None
    scene_changes = []
    frame_count = 0

    # Configure progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    status_text = st.empty()
    status_text.text("Detecting scene changes...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSV and calculate histogram
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [64, 64], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()

        if prev_hist is not None:
            # Compare histograms using correlation
            diff = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
            
            # Inverse correlation (lower = more different)
            if (1 - diff) > threshold/100:  # Normalize threshold
                scene_changes.append(frame_count / fps)
                logger.debug(f"Scene change detected at {frame_count/fps:.2f}s (diff: {1-diff:.3f})")
        
        prev_hist = hist
        frame_count += 1
        
        # Update progress
        if frame_count % 10 == 0:
            progress = min(100, int((frame_count / total_frames) * 100))
            progress_bar.progress(progress)
            status_text.text(f"Processed {frame_count}/{total_frames} frames")

    cap.release()
    progress_bar.empty()
    status_text.text(f"Detected {len(scene_changes)} scene changes")
    logger.info(f"Detected {len(scene_changes)} scene changes in {video_path}")
    
    return scene_changes

def extract_key_frames_from_section(video_path: str, 
                                  start_time: float, 
                                  end_time: float, 
                                  interval_sec: int = 5) -> List[np.ndarray]:
    """
    Extract representative frames from a specific video section.
    
    Args:
        video_path: Path to the video file
        start_time: Section start time in seconds
        end_time: Section end time in seconds
        interval_sec: Interval between extracted frames
    
    Returns:
        List of extracted frames as numpy arrays (RGB format)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Could not open video: {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    start_frame = int(start_time * fps)
    end_frame = int(end_time * fps)
    frame_interval = max(1, int(fps * interval_sec))
    
    frames = []
    frame_count = start_frame
    
    # Progress tracking
    total_frames = end_frame - start_frame
    processed_frames = 0
    
    while frame_count <= end_frame:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = cap.read()
        
        if ret:
            try:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
        
        frame_count += frame_interval
        processed_frames += frame_interval
        
    cap.release()
    logger.info(f"Extracted {len(frames)} key frames from section {start_time:.1f}s-{end_time:.1f}s")
    return frames


def analyze_ad_content(ad_path: str) -> Dict[str, Any]:
    import traceback
    """Perform comprehensive analysis of an ad video using frame sampling"""
    analysis = {
        'objects': [],
        'scene_context': '',
        'audio_transcript': '',
        'embedding': None
    }
    
    try:
        # 1. Extract key frames
        frames = extract_key_frames(ad_path)
        
        # 2. Object detection across frames
        object_set = set()
        for frame in frames:
            objects = detect_objects(frame)
            # Extract just the class names from detected objects
            class_names = extract_class_names(objects)
            object_set.update(class_names)
        analysis['objects'] = list(object_set)
    
        # 3. Scene analysis (vote from frames)
        scene_votes = {}
        for frame in frames:
            scene = analyze_scene_with_qwen(frame)
            if isinstance(scene, dict):
                scene = str(scene)
            scene_votes[scene] = scene_votes.get(scene, 0) + 1
        analysis['scene_context'] = max(scene_votes.items(), key=lambda x: x[1])[0]
    
        # 4. Audio transcription (uses full video path)
        analysis['audio_transcript'] = transcribe_audio(ad_path)
        
        # 5. Create text embedding
        text_for_embedding = f"{analysis['scene_context']} {' '.join(analysis['objects'])} {analysis['audio_transcript']}"
        analysis['embedding'] = sbert_model.encode(text_for_embedding)
    except Exception as e:
        logger.error(f"Error analyzing ad {ad_path}: {str(e)}\n{traceback.format_exc()}")
    
    return analysis

def save_complete_ad_data(ad_id: str, ad_data: Dict[str, Any], analysis_dir="assets/ad_analyses") -> bool:
    import traceback
    """Save complete ad metadata including analysis to JSON file"""
    os.makedirs(analysis_dir, exist_ok=True)
    file_path = os.path.join(analysis_dir, f"{ad_id}.json")
    
    try:
        # Make a copy to avoid modifying original data
        save_data = ad_data.copy()
        
        # Convert numpy arrays to lists for JSON serialization
        if 'analysis' in save_data and 'embedding' in save_data['analysis']:
            if isinstance(save_data['analysis']['embedding'], np.ndarray):
                save_data['analysis']['embedding'] = save_data['analysis']['embedding'].tolist()
        
        # Add timestamp for version tracking
        save_data['last_updated'] = datetime.now().isoformat()
        
        with open(file_path, 'w') as f:
            json.dump(save_data, f, indent=2)
            
        logger.info(f"Saved complete ad data for {ad_id}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save ad data for {ad_id}: {str(e)}\n{traceback.format_exc()}")
        return False

def load_complete_ad_data(ad_id: str, analysis_dir="assets/ad_analyses") -> Dict[str, Any]:
    # import traceback
    # """Load complete ad metadata from JSON file"""
    file_path = os.path.join(analysis_dir, f"{ad_id}.json")
    
    if not os.path.exists(file_path):
        logger.debug(f"No data file found for ad {ad_id}")
        return None
    
    try:
        with open(file_path, 'r') as f:
            ad_data = json.load(f)
            
            # Convert embedding back to numpy array if it exists
            if 'analysis' in ad_data and 'embedding' in ad_data['analysis']:
                ad_data['analysis']['embedding'] = np.array(ad_data['analysis']['embedding'])
            
            logger.debug(f"Loaded ad data for {ad_id}")
            return ad_data
            
    except Exception as e:
        # logger.error(f"Error loading ad data for {ad_id}: {str(e)}\n{traceback.format_exc()}")
        logger.error(f"Error loading ad data for {ad_id}: {str(e)}")
        return None

def scan_ads_by_category(ads_root="assets/ads/Brands", force_reanalyze=False) -> List[Dict[str, Any]]:
    """
    Scan categorized ad folders and return complete ad database.
    Uses cached analysis files unless force_reanalyze=True or video is newer than analysis.
    """
    ad_database = []
    
    if not os.path.exists(ads_root):
        logger.error(f"Ad root folder not found: {ads_root}")
        return []

    for category in os.listdir(ads_root):
        category_path = os.path.join(ads_root, category)
        if not os.path.isdir(category_path):
            continue

        for ad_file in os.listdir(category_path):
            if not any(ad_file.lower().endswith(fmt) for fmt in SUPPORTED_VIDEO_FORMATS):
                continue

            ad_path = os.path.join(category_path, ad_file)
            ad_id = f"{category}_{os.path.splitext(ad_file)[0]}"
            
            # Try to load existing data
            ad_data = load_complete_ad_data(ad_id)
            
            # Check if we need to reanalyze
            needs_analysis = (
                force_reanalyze or 
                not ad_data or 
                not os.path.exists(ad_path) or
                (os.path.exists(ad_path) and 
                 (not ad_data.get('last_updated') or 
                  datetime.fromisoformat(ad_data['last_updated']) < datetime.fromtimestamp(os.path.getmtime(ad_path))))
            )
            
            if needs_analysis:
                logger.info(f"Analyzing {ad_id} (force={force_reanalyze})")
                
                # Get video metadata first
                try:
                    cap = cv2.VideoCapture(ad_path)
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    duration = frame_count / fps if fps > 0 else 0
                    cap.release()
                except Exception as e:
                    logger.error(f"Error getting video info for {ad_id}: {str(e)}")
                    duration = 0
                
                # Perform analysis
                analysis = analyze_ad_content(ad_path) if os.path.exists(ad_path) else None
                
                # Create complete ad data structure
                ad_data = {
                    "id": ad_id,
                    "name": os.path.splitext(ad_file)[0].replace("_", " ").title(),
                    "path": ad_path,
                    "category": category,
                    "duration": round(duration, 2),
                    "format": "mid-roll",
                    "ctr": random.uniform(0.05, 0.2),
                    "analysis": analysis or {},
                    "last_updated": datetime.now().isoformat()
                }
                
                # Save complete data
                if not save_complete_ad_data(ad_id, ad_data):
                    logger.error(f"Failed to save data for {ad_id}, skipping")
                    continue
            
            # Add to database if valid
            if ad_data and ad_data.get('analysis'):
                ad_database.append(ad_data)
            else:
                logger.warning(f"Skipping incomplete ad data for {ad_id}")
    
    logger.info(f"Ad database loaded with {len(ad_database)} valid ads")
    return ad_database

def get_ad_database(force_refresh=False) -> List[Dict[str, Any]]:
    """
    Main interface to get ad database.
    Returns cached data unless force_refresh=True.
    """
    global _cached_ad_database
    
    if not force_refresh and '_cached_ad_database' in globals():
        return _cached_ad_database
    
    _cached_ad_database = scan_ads_by_category(force_reanalyze=force_refresh)
    return _cached_ad_database



def find_best_matching_ads_for_slot(slot: Dict[str, Any], 
                                   ad_database: List[Dict[str, Any]], 
                                   weights: Dict[str, float],
                                   top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Enhanced matching with Jaccard similarity, category matching, and semantic similarity
    """
    if not slot or not ad_database:
        return []
    
    # Prepare slot data
    slot_objects = set(obj.lower() for obj in slot['analysis'].get('objects', []))
    slot_category = slot['analysis'].get('scene_context', 'general').lower()
    slot_transcript = " ".join(slot['analysis'].get('audio', []))
    slot_embedding = sbert_model.encode(slot_transcript, convert_to_tensor=True) if slot_transcript else None
    
    scored_ads = []
    
    for ad in ad_database:
        if 'analysis' not in ad:
            continue
            
        ad_analysis = ad['analysis']
        score = 0
        match_details = {}
        
        # 1. Category (context matching) matching (exact match)
        ad_category = ad_analysis.get("scene_context", "").lower()
        if slot_category == ad_category:
            category_score = weights['category']
            match_details['category'] = category_score
            score += category_score
            match_details["context_present"] = ad_category
        
        # 2. Object matching using Jaccard similarity
        ad_objects = set(obj.lower() for obj in ad_analysis.get('objects', []))
        if slot_objects and ad_objects:
            intersection = len(slot_objects & ad_objects)
            union = len(slot_objects | ad_objects)
            jaccard_sim = intersection / union if union > 0 else 0
            object_score = weights['objects'] * min(1, len(slot_objects & ad_objects)/3)
            match_details['objects'] = object_score
            score += object_score
            match_details["object_similarity"] = jaccard_sim
        
        # 3. Semantic similarity of transcripts
        if slot_embedding is not None:
            ad_transcript = ad_analysis.get('audio_transcript', {})
            ad_transcription = ad_transcript.get('transcript', '')
            if ad_transcription:
                ad_transcript_embedding = sbert_model.encode(ad_transcription, convert_to_tensor=True)
                # Cosine similarity between embeddings
                similarity = util.pytorch_cos_sim(slot_embedding, ad_transcript_embedding).item()
                audio_score = weights['audio'] * max(0, similarity)
                match_details['audio'] = audio_score
                match_details["text_similarity"] = similarity
                score += audio_score
        
        # # 5. CTR bonus
        # ctr_score = weights['ctr'] * ad.get('ctr', 0)
        # match_details['ctr'] = ctr_score
        # score += ctr_score
        
        # Store results
        ad['match_score'] = round(score, 4)  # More precision for comparison
        ad['match_details'] = match_details
        scored_ads.append(ad)
    
    # Sort by score and return top_n
    return sorted(scored_ads, key=lambda x: x['match_score'], reverse=True)[:top_n]


def describe_best_use_case(ad: Dict[str, Any], video_analysis: Dict[str, Any], weights: Dict[str, float]) -> str:
    """
    Generates a detailed explanation of why an ad was matched, including quantitative matching scores.
    
    Args:
        ad: The matched ad dictionary (must contain 'match_details', 'name', and 'analysis')
        video_analysis: Dictionary containing video analysis results with:
            - 'objects': List of detected objects
            - 'scene_context': Scene description
            - 'audio_transcript': Transcript text
        weights: Dictionary containing the weights used for matching
        
    Returns:
        String explaining the match reasons with quantitative scores and weight percentages
    """
    reasons = []
    details = ad.get('match_details', {})
    ad_analysis = ad.get('analysis', {})
    
    # Extract comparison data
    ad_objects = set(obj.lower() for obj in ad_analysis.get('objects', []))
    video_objects = set(obj.lower() for obj in video_analysis.get('objects', []))
    # ad_category = ad.get('category', '').lower()
    
    # 1. Category Matching Explanation
    if details.get('context_present',''):
        reasons.append(
            f"category '{details.get('context_present')} is present in both video context and advertisement context'"
            )
        
    
    # 2. Object Matching Explanation (with Jaccard similarity)
    if details.get('object_similarity', 0) > 0:
        # Calculate Jaccard similarity
        intersection = ad_objects & video_objects
        union = ad_objects | video_objects
        jaccard = len(intersection) / len(union) if union else 0
        
        if intersection:
            reasons.append(
                f"shares {len(intersection)} objects ({', '.join(intersection)}) "
                f"with Jaccard similarity {jaccard:.2f} "
                f"({weights['objects']*100:.0f}% weight)"
            )
    
    # 3. Audio/Text Matching Explanation (with similarity score)
    if details.get('text_similarity', 0) > 0:
        similarity_score = details['text_similarity']
        reasons.append(
            f"audio/text similarity {similarity_score:.2f} "
            f"({weights['audio']*100:.0f}% weight)"
        )
        
        # Show top matching keywords if available
        if 'matched_keywords' in details:
            reasons[-1] += f" (keywords: {', '.join(details['matched_keywords'][:3])})"
    
    # # 4. Embedding Similarity Explanation
    # if details.get('embedding_similarity', 0) > 0:
    #     reasons.append(
    #         f"content embedding similarity {details['embedding_similarity']:.2f} "
    #         f"({weights['embedding']*100:.0f}% weight)"
    #     )
    
    # 5. CTR Bonus Explanation
    if details.get('ctr', 0) > 0:
        reasons.append(
            f"high CTR {ad['ctr']*100:.1f}% "
            f"({weights['ctr']*100:.0f}% bonus)"
        )
    
    # Format the final explanation
    if not reasons:
        return "General match (no strong matching criteria)"
    
    explanation = "Selected because:\n- " + "\n- ".join(reasons)
    
    return explanation

def show_ad_matching_interface(video_analysis: Dict[str, Any]):
    """Complete ad matching interface with slot-based recommendations"""
    st.subheader("Ad Matching & Selection")
    # Initialize session state
    if 'selected_ads' not in st.session_state:
        st.session_state.selected_ads = []
    
    # Load ad database
    ad_database = scan_ads_by_category()
    if not ad_database:
        st.warning("No ads found in the ads directory. Please upload ads first.")
        return
    
    # Get weight presets from video analysis
    weights = video_analysis.get('weight_presets', WEIGHT_PRESETS['generic'])
    
    # Slot-Based Recommendations Section
    st.header("📌 Slot-Specific Recommendations")
    
    if 'slots' not in video_analysis or not video_analysis['slots']:
        st.warning("No temporal slots available in video analysis")
    else:
        # Create tabs for each slot
        slot_tabs = st.tabs([f"Slot {slot['slot_number']} ({slot['start_time']:.1f}s-{slot['end_time']:.1f}s)" 
                           for slot in video_analysis['slots']])
        
        for idx, slot_tab in enumerate(slot_tabs):
            with slot_tab:
                slot = video_analysis['slots'][idx]
                
                # Slot metadata display
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Start Time", f"{slot['start_time']:.1f}s")
                with col2:
                    st.metric("Duration", f"{slot['duration']:.1f}s")
                with col3:
                    st.metric("Context Window", f"{slot['context_window']['start']:.1f}s-{slot['context_window']['end']:.1f}s")
                
                # Get recommendations for this slot using slot-specific analysis
                slot_ads = find_best_matching_ads_for_slot(
                    slot=slot,
                    ad_database=ad_database,
                    weights=weights,
                    top_n=10
                )
                
                if not slot_ads:
                    st.info("No strong matches found for this slot. Try manual selection below.")
                    continue
                
                # Display recommendations in a grid
                st.subheader("Recommended Ads for This Slot")
                cols = st.columns(3)
                
                for i, ad in enumerate(slot_ads[:3]):  # Show top 3
                    with cols[i % 3]:
                        with st.container(border=True):
                            # Thumbnail/Preview
                            try:
                                st.video(ad['path'])
                            except:
                                st.image(Image.new('RGB', (200, 150), color='gray'))
                            
                            # Metadata
                            st.write(f"**{ad['name']}**")
                            st.caption(f"{ad['duration']}s | {ad['category'].title()}")
                            
                            # Progress bar showing match score
                            match_score = ad.get('match_score', 0)
                            st.progress(min(1.0, match_score), 
                                      text=f"Match: {match_score:.0%}")
                            
                            # Show explanation in an expander
                            with st.expander("Why this ad?"):
                                explanation = describe_best_use_case(
                                    ad, 
                                    slot['analysis'],
                                    weights
                                )
                                st.write(explanation)
                            
                            # Selection button - now explicitly assigns to slot
                            if st.button("Select for this slot", key=f"slot_{slot['slot_number']}_ad_{ad['id']}"):
                                if ad not in st.session_state.selected_ads:
                                    # Add slot assignment to the ad
                                    ad['recommended_slot'] = slot['slot_number']
                                    st.session_state.selected_ads.append(ad)
                                    st.success(f"Added {ad['name']} to Slot {slot['slot_number']}!")
                                else:
                                    # Update slot assignment if already selected
                                    existing_ad = next(a for a in st.session_state.selected_ads if a['id'] == ad['id'])
                                    existing_ad['recommended_slot'] = slot['slot_number']
                                    st.success(f"Moved {ad['name']} to Slot {slot['slot_number']}!")

    
    # =====================================================================
    # 2. Manual Category Selection Section
    # =====================================================================
    st.header("🛒 Manual Selection")
    
    all_categories = sorted(list(set(ad['category'] for ad in ad_database)))
    selected_categories = st.multiselect(
        "Browse by category",
        options=all_categories,
        default=all_categories[:1]
    )
    
    if selected_categories:
        category_ads = [ad for ad in ad_database if ad['category'] in selected_categories]
        
        if category_ads:
            # Display as responsive grid
            cols = st.columns(3)
            for i, ad in enumerate(category_ads):
                with cols[i % 3]:
                    with st.container(border=True):
                        # Preview
                        try:
                            st.video(ad['path'], start_time=5)  # Show from 5s in
                        except:
                            st.image(Image.new('RGB', (200, 150), color='gray'))
                        
                        # Metadata
                        st.write(f"**{ad['name']}**")
                        st.caption(f"{ad['category'].title()} • {ad['duration']}s")
                        
                        # Selection button
                        if st.button("Select", key=f"manual_{ad['id']}"):
                            if ad not in st.session_state.selected_ads:
                                st.session_state.selected_ads.append(ad)
                                st.success("Added to selection!")
    
    # =====================================================================
    # 4. Selected Ads Management Section
    # =====================================================================
    st.header("🛍️ Your Ad Selection")
    
    if not st.session_state.selected_ads:
        st.info("No ads selected yet. Choose ads from above sections.")
    else:
        total_duration = sum(ad['duration'] for ad in st.session_state.selected_ads)
        video_duration = video_analysis['metadata']['duration']
        ad_ratio = total_duration / video_duration
        
        cols = st.columns(3)
        cols[0].metric("Selected Ads", len(st.session_state.selected_ads))
        cols[1].metric("Total Ad Time", f"{total_duration}s")
        cols[2].metric("Video Coverage", f"{ad_ratio:.1%}")
        
        # Display selected ads with removal option
        for i, ad in enumerate(st.session_state.selected_ads):
            with st.expander(f"{i+1}. {ad['name']}", expanded=True):
                cols = st.columns([1, 3, 1])
                with cols[0]:
                    try:
                        st.video(ad['path'])
                    except:
                        st.image(Image.new('RGB', (150, 100), color='gray'))
                
                with cols[1]:
                    st.write(f"**Category:** {ad['category'].title()}")
                    st.write(f"**Duration:** {ad['duration']}s")
                    if 'match_score' in ad:
                        st.write(f"**Match Score:** {ad['match_score']:.2f}")
                
                with cols[2]:
                    if st.button("Remove", key=f"remove_{ad['id']}"):
                        st.session_state.selected_ads.remove(ad)
                        st.rerun()
        
        # Final confirmation
        if st.button("Confirm Selection", type="primary"):
            st.session_state.ad_selection_complete = True
            st.success("Ad selection confirmed! Proceed to placement.")


def clear_frame_cache():
    """Clear all cached frames with logging"""
    logger.info("Clearing frame cache...")
    try:
        for f in os.listdir(FRAME_SAVE_DIR):
            if f.startswith("frame_") and f.endswith(".jpg"):
                os.remove(os.path.join(FRAME_SAVE_DIR, f))
        logger.info("Frame cache cleared successfully")
        st.success("Frame cache cleared!")
    except Exception as e:
        logger.error(f"Error clearing frame cache: {e}")
        st.error("Failed to clear frame cache")

def validate_frame(frame: Union[np.ndarray, Image.Image, torch.Tensor], frame_idx: int) -> Union[np.ndarray, None]:
    """Validate and normalize frame format"""
    try:
        logger.debug(f"Validating frame {frame_idx} - Type: {type(frame)}")
        
        # Convert to numpy array if needed
        if isinstance(frame, Image.Image):
            frame = np.array(frame)
            logger.debug(f"Converted PIL Image to numpy array - Shape: {frame.shape}")
        elif isinstance(frame, torch.Tensor):
            frame = frame.cpu().numpy()
            logger.debug(f"Converted Tensor to numpy array - Shape: {frame.shape}")
        
        # Check if frame is valid
        if not isinstance(frame, np.ndarray):
            logger.warning(f"Frame {frame_idx} is not a numpy array - Type: {type(frame)}")
            return None
        
        if frame.size == 0:
            logger.warning(f"Frame {frame_idx} is empty")
            return None
        
        # Convert to uint8 if needed
        if frame.dtype != np.uint8:
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
                logger.debug(f"Converted normalized frame to uint8")
            else:
                frame = frame.astype(np.uint8)
                logger.debug(f"Converted frame to uint8")
        
        # Convert BGR to RGB if needed
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            logger.debug(f"Converted BGR to RGB")
        
        logger.debug(f"Frame {frame_idx} validated - Shape: {frame.shape}, Dtype: {frame.dtype}")
        return frame
    
    except Exception as e:
        logger.error(f"Error validating frame {frame_idx}: {e}")
        return None

def extract_key_frames(video_path: str, interval_sec: int = 5) -> List[np.ndarray]:
    """Extract and cache key frames with robust error handling"""
    logger.info(f"Starting frame extraction from: {video_path}")
    frames = []
    
    try:
        # Check for existing frames first
        existing_frames = sorted([
            f for f in os.listdir(FRAME_SAVE_DIR) 
            if f.startswith("frame_") and f.endswith(".jpg")
        ])
        
        if existing_frames:
            logger.info(f"Found {len(existing_frames)} cached frames")
            for frame_file in existing_frames:
                frame_path = os.path.join(FRAME_SAVE_DIR, frame_file)
                frame = cv2.imread(frame_path)
                if frame is not None:
                    try:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(frame_rgb)
                    except Exception as e:
                        logger.error(f"Error converting cached frame {frame_file}: {e}")
            return frames
        
        # Process video if no cached frames
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Failed to open video file: {video_path}")
            return []
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_interval = max(1, int(fps * interval_sec))
        
        logger.info(f"Video properties - FPS: {fps}, Total Frames: {total_frames}, Interval: {frame_interval}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for frame_count in range(0, total_frames, frame_interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
            
            if not ret:
                logger.warning(f"Failed to read frame at position {frame_count}")
                continue
            
            logger.debug(f"Processing frame {frame_count} - Shape: {frame.shape}, Type: {type(frame)}")
            
            try:
                # Save frame
                frame_path = os.path.join(FRAME_SAVE_DIR, f"frame_{len(frames)}.jpg")
                success = cv2.imwrite(frame_path, frame)
                if not success:
                    logger.error(f"Failed to save frame {frame_count}")
                    continue
                
                # Process frame
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
                
                # Update progress
                progress = min(100, int((frame_count / total_frames) * 100))
                progress_bar.progress(progress)
                status_text.text(f"Extracted {len(frames)} frames...")
                
                logger.debug(f"Successfully processed frame {frame_count}")
            except Exception as e:
                logger.error(f"Error processing frame {frame_count}: {e}")
                continue
        
        progress_bar.empty()
        logger.info(f"Extracted {len(frames)} key frames")
        status_text.text(f"Extracted {len(frames)} key frames")
        return frames
    
    except Exception as e:
        logger.error(f"Frame extraction failed: {e}")
        return []
    finally:
        if 'cap' in locals() and cap.isOpened():
            cap.release()
            logger.info("Video capture released")

def detect_objects(frame):
    """Detect objects in frame using YOLO"""
    if not st.session_state.models_loaded or st.session_state.models is None:
        logger.warning("Models not loaded for object detection")
        return []

    try:
        # Validate frame first
        validated_frame = validate_frame(frame, 0)
        if validated_frame is None:
            logger.warning("Invalid frame provided for object detection")
            return []
        
          
        # Convert to tensor and move to device - ensure it's float32 first
        frame_tensor = torch.from_numpy(validated_frame).float() / 255.0  # Normalize here
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)
        
        # Use half precision if on GPU
        if MODEL_DEVICE == "cuda":
            frame_tensor = frame_tensor.half()
        
        frame_tensor = frame_tensor.to(MODEL_DEVICE)
        
        # Resize frame to be divisible by 32
        _, _, height, width = frame_tensor.shape
        new_height = (height // 32) * 32
        new_width = (width // 32) * 32
        frame_tensor = torch.nn.functional.interpolate(
            frame_tensor, 
            size=(new_height, new_width), 
            mode='bilinear', 
            align_corners=False
        )

        with torch.no_grad():
            results = st.session_state.models['yolo'](frame_tensor)

        objects = []
        for result in results:
            for box in result.boxes:
                obj = {
                    'class': result.names[int(box.cls)],
                    'confidence': float(box.conf),
                    'bbox': box.xywh[0].tolist()
                }
                objects.append(obj)
        # st.write(objects)
        logger.info(f"Detected {len(objects)} objects in frame")
        return objects
    except Exception as e:
        logger.error(f"Object detection failed: {e}")
        return []
    
def extract_class_names(objects):
    """
    Extracts and returns the class names from a list of detected objects.

    Args:
        objects (list): List of dictionaries containing object detection results.

    Returns:
        list: List of class names.
    """
    return [obj['class'] for obj in objects if 'class' in obj]



def transcribe_audio(audio_path):
    """Transcribe audio using Whisper"""
    if not st.session_state.models_loaded or st.session_state.models is None:
        logger.warning("Models not loaded for audio transcription")
        return {"transcript": "[Models not loaded]", "language": "unknown"}
    
    if 'whisper' not in st.session_state.models or st.session_state.models['whisper'] is None:
        logger.warning("Whisper model not available")
        return {"transcript": "[Whisper model not available]", "language": "unknown"}
    
    try:
        with torch.no_grad():
            logger.info("Starting audio transcription")
            result = st.session_state.models['whisper'].transcribe(audio_path)
        return {
            "transcript": result.get('text', ''),
            "language": result.get('language', 'unknown')
        }
    except Exception as e:
        logger.error(f"Audio transcription failed: {e}")
        return {"transcript": "[Transcription failed]", "language": "unknown"}
    

def extract_audio_from_video(video_path):
    """Extract audio track from video file"""
    logger.info(f"Extracting audio from video: {video_path}")
    try:
        audio_dir = tempfile.mkdtemp()
        audio_path = os.path.join(audio_dir, "audio.wav")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        command = f"ffmpeg -i \"{video_path}\" -q:a 0 -map a \"{audio_path}\" -y"
        result = os.system(command)

        if result != 0 or not os.path.exists(audio_path):
            raise RuntimeError(f"Failed to extract audio from video: {video_path}")

        logger.info("Audio extraction successful")
        return audio_path
    except Exception as e:
        logger.error(f"Audio extraction failed: {e}")
        return None

def extract_audio_segment_from_video(video_path, start_time, end_time, output_format="wav"):
    """
    Extract a specific audio segment from video file
    
    Args:
        video_path (str): Path to the video file
        start_time (float): Start time in seconds
        end_time (float): End time in seconds
        output_format (str): Output audio format (wav, mp3, etc.)
        
    Returns:
        str: Path to the extracted audio segment or None if failed
    """
    logger.info(f"Extracting audio segment from {start_time}s to {end_time}s in video: {video_path}")
    
    try:
        # Validate input
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
            
        if end_time <= start_time:
            raise ValueError("End time must be after start time")
        
        # Create temp directory for output
        audio_dir = tempfile.mkdtemp()
        output_filename = f"audio_{start_time}_{end_time}.{output_format}"
        audio_path = os.path.join(audio_dir, output_filename)
        
        # Build ffmpeg command
        duration = end_time - start_time
        command = [
            "ffmpeg",
            "-ss", str(start_time),  # Start time
            "-i", video_path,       # Input file
            "-t", str(duration),   # Duration to extract
            "-q:a", "0",           # Audio quality (0=best)
            "-map", "a",           # Extract audio only
            "-y",                  # Overwrite output
            audio_path             # Output file
        ]
        
        # Convert command list to string for logging
        cmd_str = " ".join(command)
        logger.debug(f"Running command: {cmd_str}")
        
        # Execute command
        result = subprocess.run(command, capture_output=True, text=True)
        
        if result.returncode != 0:
            logger.error(f"FFmpeg failed with error: {result.stderr}")
            raise RuntimeError(f"Audio extraction failed: {result.stderr}")
            
        if not os.path.exists(audio_path):
            raise RuntimeError("Output file not created")
            
        logger.info(f"Audio segment extracted successfully to {audio_path}")
        return audio_path
        
    except Exception as e:
        logger.error(f"Audio segment extraction failed: {str(e)}")
        # Clean up temp directory if creation failed
        if 'audio_dir' in locals() and os.path.exists(audio_dir):
            try:
                shutil.rmtree(audio_dir)
            except Exception as clean_error:
                logger.error(f"Failed to clean temp directory: {clean_error}")
        return None

def analyze_scene_with_qwen(frame: np.ndarray) -> str:
    """
    Analyze a single frame with Qwen-VL to determine scene context.
    """
    if not st.session_state.models_loaded:
        logger.warning("Models not loaded, initializing Qwen-VL...")
        models_to_load = {'qwen_vl': True}
        st.session_state.models = load_selected_models(models_to_load)
        st.session_state.models_loaded = True
    
    if 'qwen_vl' not in st.session_state.models or st.session_state.models['qwen_vl'] is None:
        logger.warning("Qwen-VL model not available, falling back to generic category")
        return "general"
    
    try:
        # Convert frame to PIL Image
        pil_image = Image.fromarray(frame)
        
        # Create temp file for Qwen processing
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp:
            pil_image.save(tmp.name)
            tmp_path = os.path.abspath(tmp.name)
            
            # Query Qwen with retry logic
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    prompt = "What is the main category of this scene? Choose from: sports, technology, fashion, food, or general."
                    query = st.session_state.models['qwen_tokenizer'].from_list_format([
                        {'image': tmp_path},
                        {'text': prompt},
                    ])
                    response = st.session_state.models['qwen_vl'].chat(
                        st.session_state.models['qwen_tokenizer'],
                        query=query,
                        history=[]
                    )[0]
                    # st.write(response)
                    if response:
                        # Parse response for category
                        response_lower = response
                        categories = ['sports', 'technology', 'fashion', 'food']
                        for category in categories:
                            if category in response_lower:
                                logger.info(f"Scene analyzed as: {category}")
                                return category
                        
                        logger.info("No specific category detected, using 'general'")
                        return "general"
                        
                except Exception as e:
                    if attempt < max_retries - 1:
                        logger.warning(f"Retry {attempt + 1}/{max_retries} failed: {e}")
                        time.sleep(1)  # Wait before retry
                    else:
                        logger.error(f"Scene analysis failed after {max_retries} attempts: {e}")
                        return "general"
    
    except Exception as e:
        logger.error(f"Scene analysis error: {e}")
        return "general"
    
def video_uploader(selected_video_path):
    """Handle video selection from predefined paths"""
    logger.info(f"Loading selected video: {selected_video_path}")
    
    if not os.path.exists(selected_video_path):
        logger.error(f"Selected video not found at: {selected_video_path}")
        st.error(f"Selected video not found at: {selected_video_path}")
        return None
    
    st.session_state['video_path'] = selected_video_path
    st.session_state['video_name'] = os.path.basename(selected_video_path)
    logger.info(f"Video loaded successfully: {selected_video_path}")
    return selected_video_path


def display_video(video_path):
    """Display video preview and metadata"""
    logger.info(f"Displaying video: {video_path}")
    
    st.header("Video Preview")
    col1, col2 = st.columns([3, 1])
    
    with col1:
        try:
            st.video(video_path)
            logger.debug("Video preview displayed successfully")
        except Exception as e:
            logger.error(f"Error displaying video: {e}")
            st.error("Could not display video preview")
    
    with col2:
        st.write("### Video Info")
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            cap.release()
            
            st.write(f"**Filename:** {st.session_state.video_name}")
            st.write(f"**Duration:** {duration:.2f} seconds")
            st.write(f"**FPS:** {fps:.2f}")
            st.write(f"**Frames:** {frame_count}")
            logger.debug("Video metadata displayed successfully")
        except Exception as e:
            logger.error(f"Error reading video info: {e}")
            st.error("Could not read video metadata")


def processing_options():
    """Display and collect processing options with model selection"""
    logger.info("Displaying processing options")
    
    st.header("⚙️ Processing Options")
    
    # Model Selection Section
    with st.expander("🔍 Select Analysis Models", expanded=True):
        st.write("Choose which models to use for analysis (select at least one)")
        
        # Initialize default selections in session state if not exists
        if 'selected_models' not in st.session_state:
            st.session_state.selected_models = {
                'object_detection': True,
                'scene_analysis': True,
                'audio_analysis': True,
                
            }
        
        # Model checkboxes
        col1, col2 = st.columns(2)
        with col1:
            object_detection = st.checkbox(
                "Object Detection (YOLOv8)", 
                value=st.session_state.selected_models['object_detection'],
                key='object_detection_check'
            )
            scene_analysis = st.checkbox(
                "Scene Analysis (Qwen-VL)", 
                value=st.session_state.selected_models['scene_analysis'],
                key='scene_analysis_check'
            )
        with col2:
            audio_analysis = st.checkbox(
                "Audio Analysis (Whisper)", 
                value=st.session_state.selected_models['audio_analysis'],
                key='audio_analysis_check'
            )
        
        # Update session state
        st.session_state.selected_models = {
            'object_detection': object_detection,
            'scene_analysis': scene_analysis,
            'audio_analysis': audio_analysis
        }
        
        # Warn if no models selected
        if not any(st.session_state.selected_models.values()):
            st.warning("Please select at least one analysis model")
    
    # Frame Sampling Options
    with st.expander("🎞️ Frame Sampling", expanded=False):
        frame_interval = st.slider(
            "Frame Analysis Interval (seconds)", 
            1, 10, 5,
            help="How often to sample frames for analysis"
        )

    options = {
        "analyze_objects": st.session_state.selected_models['object_detection'],
        "analyze_scenes": st.session_state.selected_models['scene_analysis'],
        "analyze_audio": st.session_state.selected_models['audio_analysis'],
        "frame_interval": frame_interval
    }
    
    logger.info(f"Processing options selected: {options}")
    return options

@st.cache_resource(show_spinner="Loading selected AI models...")
def load_selected_models(models_to_load):
    """Load only the selected models to save memory"""
    models = {}
    device = MODEL_DEVICE
    
    try:
        logger.info(f"Loading selected models: {models_to_load}")
        cudnn.benchmark = True
        
        if models_to_load.get('yolo', False):
            logger.info("Loading YOLOv8 model...")
            models['yolo'] = YOLO('yolov8n.pt').to(device)
            if device == "cuda":
                models['yolo'] = models['yolo'].half()
            models['yolo'].eval()
        
        if models_to_load.get('whisper', False):
            logger.info("Loading Whisper model...")
            try:
                models['whisper'] = whisper.load_model("base", device=device)
            except Exception as e:
                logger.error(f"Whisper loading failed: {e}")
                models['whisper'] = None
        
        if models_to_load.get('qwen_vl', False):
            logger.info("Loading Qwen-VL-Chat...")
            try:
                models['qwen_tokenizer'] = AutoTokenizer.from_pretrained(
                    "Qwen/Qwen-VL-Chat",
                    trust_remote_code=True
                )
                models['qwen_vl'] = AutoModelForCausalLM.from_pretrained(
                    "Qwen/Qwen-VL-Chat",
                    device_map="auto",
                    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                    trust_remote_code=True
                ).eval()
            except Exception as e:
                logger.error(f"Qwen-VL loading failed: {str(e)}")
                models['qwen_vl'] = None
        
        logger.info("Selected models loaded successfully!")
        if "models" not in st.session_state:
            st.session_state.models = None
        st.session_state.models = models
        return models
        
    except Exception as e:
        logger.error(f"Model loading failed: {str(e)}")
        torch.cuda.empty_cache()
        return None

def process_video(video_path, options, slot_times=None):
    """Main video processing pipeline with slot-based analysis and embedding cache"""
    logger.info(f"Starting video processing with options: {options}")
    
    try:
        # Initialize models if not loaded
        if not st.session_state.models_loaded:
            with st.spinner("Loading selected AI models..."):
                # Only load models that are selected
                models_to_load = {}
                if options['analyze_objects']:
                    models_to_load['yolo'] = True
                if options['analyze_scenes'] or options['analyze_content_type']:
                    models_to_load['qwen_vl'] = True
                    models_to_load['qwen_tokenizer'] = True
                if options['analyze_audio']:
                    models_to_load['whisper'] = True
                
                st.session_state.models = load_selected_models(models_to_load)
                if st.session_state.models is None:
                    logger.error("Failed to load models")
                    st.error("Failed to load models")
                    return None
                st.session_state.models_loaded = True
        
        # Extract video metadata
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0
        cap.release()
        
        video_analysis = {
            "options": options,
            "metadata": {
                "fps": fps,
                "frame_count": frame_count,
                "duration": duration,
                "video_path": video_path
            },
            "slots": []
        }
        
        # If no slot times provided, analyze entire video as one slot
        if slot_times is None:
            slot_times = [{
                "slot_number": 1,
                "start": 0,
                "end": duration,
                "context_start": 0,
                "context_end": duration
            }]
        
        progress_bar = st.progress(0)
        total_slots = len(slot_times)
        
        for i, slot in enumerate(slot_times):
            progress = (i + 1) / total_slots
            progress_text = f"Processing Slot {slot['slot_number']} ({i+1}/{total_slots})"
            progress_bar.progress(progress, text=progress_text)

            with st.expander(f"🔍 Slot {slot['slot_number']} ({slot['context_start']//60}:{slot['context_start']%60:02d}-{slot['context_end']//60}:{slot['context_end']%60:02d})"):
                try:
                    slot_analysis = {
                        "slot_number": slot['slot_number'],
                        "start_time": slot['start'],
                        "end_time": slot['end'],
                        "duration": slot['end'] - slot['start'],
                        "context_window": {
                            "start": slot['context_start'],
                            "end": slot['context_end']
                        },
                        "analysis": {
                            'objects': [],
                            'scene_context': [],
                            'audio': [],
                            'embedding': []
                        }
                    }
                    
                    # Extract frames from context window
                    frames = extract_key_frames_from_section(
                        video_path,
                        slot['context_start'],
                        slot['context_end'],
                        options['frame_interval']
                    )
                    
                    if not frames:
                        st.warning(f"No frames extracted for slot {slot['slot_number']}")
                        continue
                    
                    # Object detection
                    if options['analyze_objects']:
                        with st.spinner("Detecting objects..."):
                            all_objects = []
                            for frame in frames:
                                all_objects.extend(detect_objects(frame))
                            # st.write(f"All objects: {all_objects}")
                            obj_counts = Counter(obj['class'] for obj in all_objects)
                            slot_analysis['analysis']['objects'] = [obj for obj, _ in obj_counts.most_common(5)]
                            st.write(f"Top objects: {', '.join(slot_analysis['analysis']['objects'])}")
                    
                    # Scene analysis
                    if options['analyze_scenes'] and 'qwen_vl' in st.session_state.models:
                        with st.spinner("Analyzing scene..."):
                            mid_frame = frames[len(frames) // 2]
                            slot_analysis['analysis']['scene_context'] = analyze_scene_with_qwen(mid_frame)
                            st.write(f"Scene context: {slot_analysis['analysis']['scene_context']}")
                    
                    # Audio analysis
                    if options['analyze_audio'] and 'whisper' in st.session_state.models:
                        with st.spinner("Analyzing audio..."):
                            audio_path = extract_audio_segment_from_video(
                                video_path,
                                slot['context_start'],
                                slot['context_end']
                            )
                            if audio_path and os.path.exists(audio_path):
                                audio_result = transcribe_audio(audio_path)
                                # st.write(audio_result)
                                final_result ={
                                    'transcript': audio_result.get('transcript', ''),
                                    'language': audio_result.get('language', '')
                                    # 'segments': audio_result.get('segments', [])
                                }
                                slot_analysis['analysis']['audio'].append(final_result['transcript'])
                                transcribe_str = "".join(slot_analysis["analysis"]["audio"])
                                os.remove(audio_path)
                                st.write(f"Audio transcript: {transcribe_str[:200]}")

                    # # Generate embedding for this slot
                    # with st.spinner("Generating embedding..."):
                    #     slot_embedding = st.session_state.recommender.get_video_embedding(
                    #         f"{video_path}_slot_{slot['slot_number']}",
                    #         frames
                    #     )
                    #     slot_embedding = slot_embedding.tolist() if hasattr(slot_embedding, 'tolist') else slot_embedding
                    #     slot_analysis['analysis']['embedding'].append(slot_embedding)
                    # # st.write(slot_analysis)
                    video_analysis['slots'].append(slot_analysis)
                    
                except Exception as e:
                    logger.error(f"Error processing slot {slot['slot_number']}: {e}")
                    st.error(f"Failed to process slot {slot['slot_number']}")
                    continue
        
        progress_bar.empty()
        
        if not video_analysis['slots']:
            st.error("No slots were successfully analyzed")
            return None
        
        # Store analysis in session state
        st.session_state['video_analysis'] = video_analysis
        logger.info("Video analysis completed successfully")   
        st.success("✅ Video analysis complete!")
        return video_analysis

    except Exception as e:
        logger.error(f"Error in video processing: {e}")
        st.error("An error occurred during video processing")
        return None


def detect_video_content_type(video_analysis: Dict[str, Any]) -> str:
    """
    Detect the type of video content based on analysis results.
    Returns: 'broadcast', 'product_review', or 'generic'
    """
    # Get analysis data with fallbacks
    objects = video_analysis.get('key_objects', [])
    audio_transcript = video_analysis.get('audio', {}).get('transcript', '').lower()
    scene_context = video_analysis.get('scene_context', 'generic').lower()
    
    # Broadcast detection criteria
    broadcast_keywords = ['news', 'report', 'anchor', 'studio', 'broadcast']
    if (any(kw in audio_transcript for kw in broadcast_keywords) or \
       any(obj in ['microphone', 'news ticker', 'studio lights'] for obj in objects)):
        return 'broadcast'
    
    # Product review detection criteria
    review_keywords = ['review', 'unboxing', 'unbox', 'test', 'compare', 'versus']
    product_keywords = ['product', 'item', 'model', 'version', 'features']
    if (any(kw in audio_transcript for kw in review_keywords) or \
       (any(kw in audio_transcript for kw in product_keywords) and \
       len(objects) > 0 and 'person' in objects)):
        return 'product_review'
    
    # Default to generic
    return 'generic'

def detect_video_type(video_analysis):
    """Determine the most appropriate weight preset based on video content"""
    if video_analysis is None:
        return "generic"
    
    content_type = video_analysis.get('content_type', 'generic')
    
    # For broadcast and product review, use their specific presets
    if content_type in ['broadcast', 'product_review']:
        return content_type
    
    # Fall back to audio/visual detection for generic content
    audio_score = len(video_analysis.get('audio', {}).get('transcript', '').split())
    object_score = len(video_analysis.get('key_objects', []))
    
    if audio_score > 50 and object_score < 5:  # Podcast/Interview
        return "audio_heavy"
    elif object_score >= 10 and audio_score < 20:  # Unboxing/Sports
        return "visual_heavy"
    else:  # Generic
        return "generic"


def update_weight_presets(selected_type: str) -> Dict[str, float]:
    """
    Update and return weight presets based on selected content type and analysis
    
    Args:
        selected_type: The type of content detected/selected
        
    Returns:
        Dictionary containing updated weight values for different matching criteria
    """
    base_presets = {
        "broadcast": {
            "category": 0.3,    # Lower category importance for broad audience
            "objects": 0.2,     # Lower object importance
            "audio": 0.4,       # Higher audio importance for broadcast content
            "ctr": 0.1         # Higher CTR importance for broad reach
        },
        "product_review": {
            "category": 0.5,    # Highest category importance for targeted matching
            "objects": 0.35,    # High object importance for product relevance
            "audio": 0.1,      # Lower audio importance
            "ctr": 0.05        # Lower CTR importance for targeted audience
        },
        "generic": {
            "category": 0.35,   # Balanced category importance
            "objects": 0.3,     # Moderate object importance
            "audio": 0.25,      # Moderate audio importance
            "ctr": 0.1         # Standard CTR importance
        }
    }
    
    # Return the preset for the selected type, fallback to generic if not found
    return base_presets.get(selected_type, base_presets["generic"])

def video_content_analysis_interface(video_analysis: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    """
    Display interface for video content type selection and visualization.
    Returns tuple of (selected_content_type, analysis_data)
    """
    # st.header("📺 Video Content Analysis")
    
    # 1. Automatic content type detection
    detected_types = {
        'broadcast': 0,
        'product_review': 0,
        'generic': 0
    }
    
    # Calculate detection scores (simplified example)
    objects = video_analysis.get('key_objects', [])
    audio_text = video_analysis.get('audio', {}).get('transcript', '').lower()
    
    # Broadcast detection score
    broadcast_keywords = ['news', 'report', 'anchor', 'studio', 'broadcast']
    detected_types['broadcast'] = sum(audio_text.count(kw) for kw in broadcast_keywords)
    detected_types['broadcast'] += sum(1 for obj in objects if obj in ['microphone', 'news ticker', 'studio lights'])
    
    # Product review detection score
    review_keywords = ['review', 'unboxing', 'unbox', 'test', 'compare', 'versus']
    product_keywords = ['product', 'item', 'model', 'version', 'features']
    detected_types['product_review'] = sum(audio_text.count(kw) for kw in review_keywords + product_keywords)
    detected_types['product_review'] += sum(1 for obj in objects if obj in ['product', 'box', 'package'])
    
    # Generic is the baseline
    detected_types['generic'] = max(1, len(objects) + len(audio_text.split()) // 10)
    
    # Normalize scores
    total = sum(detected_types.values())
    if total > 0:
        for k in detected_types:
            detected_types[k] = detected_types[k] / total
    
    selected_type = "generic"
    default_type = max(detected_types, key=detected_types.get)
    # Store the detection data in video analysis
    video_analysis['content_type'] = {
        'selected': selected_type,
        'detected': detected_types,
        'auto_suggestion': default_type
    }
    
    return selected_type, video_analysis
        

def generate_final_video(video_path: str, 
                       segments: List[Dict], 
                       output_format: str, 
                       quality: int, 
                       transitions: bool) -> str:
    """Generate final video with ads integrated using FFmpeg"""
    
    # 1. Validate inputs and setup paths
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Input video not found: {video_path}")
    
    output_dir = os.path.abspath("outputs/processed_videos")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"final_video_{int(time.time())}.{output_format}")

    # 2. Create temp directory for processing
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # 3. Process each segment
            segment_files = []
            for i, segment in enumerate(segments):
                segment_file = os.path.join(temp_dir, f"segment_{i}.mp4")
                
                if segment['type'] == 'video':
                    # Build FFmpeg command for video segment
                    cmd = [
                        'ffmpeg', '-y',
                        '-ss', str(segment['start']),
                        '-i', video_path,
                        '-to', str(segment['end']),
                        '-c:v', 'libx264',
                        '-crf', str(31 - quality * 3),  # Quality mapping
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        segment_file
                    ]
                else:
                    # Verify ad file exists
                    if not os.path.exists(segment['ad']['path']):
                        raise FileNotFoundError(f"Ad file not found: {segment['ad']['path']}")
                    
                    # Build FFmpeg command for ad segment
                    cmd = [
                        'ffmpeg', '-y',
                        '-i', segment['ad']['path'],
                        '-c:v', 'libx264',
                        '-crf', str(31 - quality * 3),
                        '-preset', 'fast',
                        '-c:a', 'aac',
                        segment_file
                    ]

                # Run FFmpeg command
                try:
                    subprocess.run(cmd, check=True, 
                                  stdout=subprocess.PIPE, 
                                  stderr=subprocess.PIPE)
                    segment_files.append(segment_file)
                except subprocess.CalledProcessError as e:
                    raise RuntimeError(f"FFmpeg command failed: {e.stderr.decode('utf-8')}")

            # 4. Create concatenation list file
            list_file = os.path.join(temp_dir, "input_list.txt")
            with open(list_file, 'w') as f:
                for file in segment_files:
                    f.write(f"file '{file}'\n")

            # 5. Build final FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-f', 'concat',
                '-safe', '0',
                '-i', list_file,
                '-c:v', 'libx264',
                '-cr aac',
                output_path
            ]

            # 6. Add transitions if enabled
            if transitions and len(segment_files) > 1:
                filter_complex = []
                inputs = []
                for i, file in enumerate(segment_files):
                    inputs.extend(['-i', file])
                    if i < len(segment_files)-1:
                        filter_complex.append(
                            f"[{i}:v][{i+1}:v]xfade=transition=fade:duration=0.5:"
                            f"offset={sum(s['duration'] for s in segments[:i+1]) - 0.5}[v{i}];"
                        )
                
                if filter_complex:
                    filter_complex_str = "".join(filter_complex)
                    filter_complex_str += f"concat=n={len(segment_files)-1}:v=1:a=1 [v]"
                    cmd = inputs + [
                        '-filter_complex', filter_complex_str,
                        '-map', '[v]',
                        '-map', '0:a?',
                        output_path
                    ]

            # 7. Run final FFmpeg command
            try:
                subprocess.run(cmd, check=True)
                if not os.path.exists(output_path):
                    raise RuntimeError("Output file was not created")
                return output_path
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"Final assembly failed: {e.stderr.decode('utf-8')}")

        except Exception as e:
            if os.path.exists(output_path):
                os.remove(output_path)
            raise

# ==========
# new video_page
# ==========
def video_processing_page():
    st.header("Smart Placement - Contextual Media Matching")

    # Constants (all in seconds)
    SLOT_DURATION = 120  # 2 minutes
    MIN_CONTEXT_WINDOW = 120
    MAX_CONTEXT_WINDOW = 300
    DEFAULT_CONTEXT_WINDOW = 300

    # Validate and set up video directory
    folder_path = 'assets/sample_videos/'
    if not os.path.exists(folder_path):
        st.error(f"Video directory not found: {folder_path}")
        return

    try:
        files = os.listdir(folder_path)
        video_files = [file for file in files if file.lower().endswith(('.mkv', '.mp4'))]
        if not video_files:
            st.warning("No video files found in the directory.")
            return

        video_names = sorted(list(set(os.path.splitext(f)[0] for f in video_files)))
        selected_name = st.selectbox('Select a video', video_names, index=0)
        selected_file = next((f for f in video_files if os.path.splitext(f)[0] == selected_name), None)

        if not selected_file:
            st.error("Selected video file not found.")
            return

        video_path = os.path.join(folder_path, selected_file)
        st.session_state.video_path = video_path

        # Display video
        display_video(video_path)
        options = processing_options()

        # Get video duration
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = frame_count / fps if fps > 0 else 0
        cap.release()

        # Display video duration
        duration_display = seconds_to_hhmmss(duration_sec)

        # ---- Input Form ---- #
        with st.form("slot_form"):
            st.subheader("Enter Ad Slot Timings (HH:MM format)")
            st.caption(f"Video Duration: {duration_display} ({duration_sec:.0f} seconds)")

            context_window_minutes = st.slider(
                "Context Window (minutes before/after slot)",
                min_value = MIN_CONTEXT_WINDOW // 60,
                max_value = MAX_CONTEXT_WINDOW // 60,
                value = DEFAULT_CONTEXT_WINDOW // 60,
                step=1,
                help="Select how many minutes of content to analyze before and after each ad slot"
            )
            context_window_sec = context_window_minutes * 60

            col1, col2 = st.columns(2)
            with col1:
                slot_1_input = st.text_input("Slot 1 Start Time (HH:MM)", value="00:00")
            with col2:
                slot_2_input = st.text_input("Slot 2 Start Time (HH:MM)", value="00:02")

            submitted = st.form_submit_button("Start Processing")

        if submitted:
            # ---- Convert HH:MM to seconds ---- #
            def hhmm_to_seconds(time_str):
                try:
                    parts = time_str.strip().split(':')
                    if len(parts) != 2:
                        raise ValueError
                    hours, minutes = map(int, parts)
                    return hours * 3600 + minutes * 60
                except:
                    st.error(f"Invalid time format: '{time_str}'. Use HH:MM (e.g., 01:30 for 1 hour 30 minutes).")
                    return None

            slot1_start_sec = hhmm_to_seconds(slot_1_input)
            slot2_start_sec = hhmm_to_seconds(slot_2_input)

            if None in [slot1_start_sec, slot2_start_sec]:
                st.stop()

            slot1_end_sec = slot1_start_sec + SLOT_DURATION
            slot2_end_sec = slot2_start_sec + SLOT_DURATION

            # # Optional debug output
            # st.write("⏱️ Converted Slot Timings (seconds)", {
            #     "Slot 1 Start": slot1_start_sec,
            #     "Slot 1 End": slot1_end_sec,
            #     "Slot 2 Start": slot2_start_sec,
            #     "Slot 2 End": slot2_end_sec,
            #     "Video Duration": duration_sec,
            #     "Gap Between Slots": slot2_start_sec - slot1_end_sec,
            # })

            # ---- Validations ---- #
            if slot1_start_sec == slot2_start_sec:
                st.error("Slot 1 and Slot 2 cannot be the same.")
                st.stop()
            elif slot1_start_sec > slot2_start_sec:
                st.error("Slot 1 must be before Slot 2.")
                st.stop()
            elif slot1_end_sec > duration_sec:
                st.error(f"Slot 1 (ends at {seconds_to_hhmmss(slot1_end_sec)}) exceeds video duration ({duration_display}).")
                st.stop()
            elif slot2_end_sec > duration_sec:
                st.error(f"Slot 2 (ends at {seconds_to_hhmmss(slot2_end_sec)}) exceeds video duration ({duration_display}).")
                st.stop()
            elif (slot2_start_sec - slot1_end_sec) < SLOT_DURATION:
                st.error(f"Slots must be at least {SLOT_DURATION // 60} minutes apart.")
                st.stop()

            # ---- Slot + Context Calculation ---- #
            slot_times = [
                {
                    "slot_number": 1,
                    "start": slot1_start_sec,
                    "end": slot1_end_sec,
                    "context_start": max(0, slot1_start_sec - context_window_sec),
                    "context_end": min(duration_sec, slot1_end_sec + context_window_sec)
                },
                {
                    "slot_number": 2,
                    "start": slot2_start_sec,
                    "end": slot2_end_sec,
                    "context_start": max(0, slot2_start_sec - context_window_sec),
                    "context_end": min(duration_sec, slot2_end_sec + context_window_sec)
                }
            ]

            # ---- Run Processing ---- #
            with st.spinner("Processing video slots..."):
                video_analysis = process_video(video_path, options, slot_times)

                if video_analysis:
                    selected_type, video_analysis = video_content_analysis_interface(video_analysis)
                    video_analysis['weight_presets'] = update_weight_presets(selected_type)

                    st.session_state.video_analysis = video_analysis
                    st.success("Slot analysis complete!")
                    st.info("Proceed to Context2Content Integration for smart ad matching")

    except Exception as e:
        logger.error(f"Error in video processing page: {str(e)}")
        st.error(f"An error occurred: {str(e)}")

def seconds_to_hhmmss(seconds):
    """Convert seconds to HH:MM:SS format"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def insert_video_segment(main_video_path: str, insert_video_path: str, start_time: float, output_path: str, mute_insert: bool = False):
    """
    Insert a video segment into the main video at a specified start time using FFmpeg Python library.
    Handles different resolutions by scaling the inserted video to match the main video.
    
    Args:
        main_video_path: Path to the main video file
        insert_video_path: Path to the video segment to be inserted
        start_time: Time in seconds where the segment should be inserted
        output_path: Path for the output video file
        mute_insert: Whether to mute the inserted video segment
        
    Returns:
        Path to the output video file
    """
    try:
        # Get video information
        main_width, main_height, main_duration = get_video_info(main_video_path)
        insert_width, insert_height, insert_duration = get_video_info(insert_video_path)
        
        # Validate start time
        if start_time < 0:
            raise ValueError("Start time cannot be negative")
        if start_time > main_duration:
            raise ValueError(f"Start time {start_time}s exceeds main video duration {main_duration}s")
        
        # Create output directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Create temporary files
        temp_dir = tempfile.mkdtemp()
        temp_files = {
            'part1': os.path.join(temp_dir, 'part1_temp.mp4'),
            'part2': os.path.join(temp_dir, 'part2_temp.mp4'),
            'insert': os.path.join(temp_dir, 'insert_temp.mp4'),
            'concat_list': os.path.join(temp_dir, 'concat_list.txt')
        }
        
        # Extract part 1 (before insertion point)
        part1_duration = start_time
        if part1_duration > 0:  # Only extract if there's content before the insertion point
            (
                ffmpeg
                .input(main_video_path)
                .output(
                    temp_files['part1'],
                    vcodec='libx264',
                    acodec='aac',
                    vf=f'scale={main_width}:{main_height}',
                    ss=0,
                    t=part1_duration,
                    strict='experimental'
                )
                .run(overwrite_output=True, quiet=True)
            )
        
        # Extract part 2 (after insertion point)
        part2_start = start_time
        part2_duration = main_duration - start_time
        if part2_duration > 0:  # Only extract if there's content after the insertion point
            (
                ffmpeg
                .input(main_video_path)
                .output(
                    temp_files['part2'],
                    vcodec='libx264',
                    acodec='aac',
                    vf=f'scale={main_width}:{main_height}',
                    ss=part2_start,
                    t=part2_duration,
                    strict='experimental'
                )
                .run(overwrite_output=True, quiet=True)
            )
        
        # Process insert video (scale and optionally mute)
        insert_input = ffmpeg.input(insert_video_path)
        insert_output_args = {
            'vcodec': 'libx264',
            'acodec': 'aac',
            'vf': f'scale={main_width}:{main_height}',
            't': insert_duration,  # Ensure we don't exceed the ad duration
            'strict': 'experimental'
        }
        if mute_insert:
            insert_output_args['an'] = None
            
        (
            insert_input
            .output(temp_files['insert'], **insert_output_args)
            .run(overwrite_output=True, quiet=True)
        )
        
        # Create concatenation list
        with open(temp_files['concat_list'], 'w') as f:
            if os.path.exists(temp_files['part1']) and os.path.getsize(temp_files['part1']) > 0:
                f.write(f"file '{temp_files['part1']}'\n")
            f.write(f"file '{temp_files['insert']}'\n")
            if os.path.exists(temp_files['part2']) and os.path.getsize(temp_files['part2']) > 0:
                f.write(f"file '{temp_files['part2']}'\n")
        
        # Concatenate all parts
        (
            ffmpeg
            .input(temp_files['concat_list'], format='concat', safe=0)
            .output(
                output_path,
                vcodec='libx264',
                acodec='aac',
                strict='experimental'
            )
            .run(overwrite_output=True, quiet=True)
        )
        
        return output_path
        
    except ffmpeg.Error as e:
        raise RuntimeError(f"FFmpeg error: {e.stderr.decode('utf-8')}")
    except Exception as e:
        raise RuntimeError(f"Error inserting video segment: {str(e)}")
    finally:
        # Clean up temporary files
        if 'temp_dir' in locals():
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

def get_video_info(video_path):
    """Get video width, height and duration using ffmpeg probe"""
    try:
        probe = ffmpeg.probe(video_path)
        video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
        if not video_stream:
            raise ValueError('No video stream found')
            
        width = int(video_stream['width'])
        height = int(video_stream['height'])
        duration = float(probe['format']['duration'])
        
        return width, height, duration
        
    except Exception as e:
        raise RuntimeError(f"Could not get video info for {video_path}: {str(e)}")

def ad_integration_page():
    SLOT_DURATION = 120 # 2 minutes
    st.subheader("Ad Integration & Placement")
    
    # Initialize session state variables
    if 'selected_ads' not in st.session_state:
        st.session_state.selected_ads = []
    if 'ad_placements' not in st.session_state:
        st.session_state.ad_placements = {}
    if 'frame_cache' not in st.session_state:
        st.session_state.frame_cache = {}
    
    # Create tabs
    tab1, tab2 = st.tabs(["Ad Integration", "Placement & Preview"])
    
    with tab1:
        if 'video_analysis' not in st.session_state or not st.session_state.video_analysis:
            st.warning("Please complete video processing first")
        else:
            show_ad_matching_interface(st.session_state.video_analysis)

    with tab2:
        st.subheader("Ad Placement Configuration & Live Preview")
        
        if not st.session_state.selected_ads:
            st.warning("No ads selected yet. Please select ads from the Ad Integration tab.")
            return
            
        # Get available slots from video analysis
        if 'video_analysis' not in st.session_state or not st.session_state.video_analysis:
            st.error("No ad slots available. Please process the video first.")
            return
            
        slots = sorted(st.session_state.video_analysis['slots'], key=lambda x: x['slot_number'])
        SLOT_DURATION = 120  # 2 minutes in seconds
        
        # Placement configuration section
        st.subheader("Slot-based Placement")
        
        # Group selected ads by their recommended slots
        slot_ads = {slot['slot_number']: [] for slot in slots}
        for ad in st.session_state.selected_ads:
            if 'recommended_slot' in ad:
                if ad['recommended_slot'] in slot_ads:
                    slot_ads[ad['recommended_slot']].append(ad)
                else:
                    st.warning(f"Ad '{ad['name']}' has invalid slot number {ad['recommended_slot']}")
        
        # Display placement for each slot
        for slot in slots:
            slot_number = slot['slot_number']
            slot_start = slot['start_time']
            slot_end = slot_start + SLOT_DURATION
            
            with st.expander(f"Slot {slot_number} ({slot_start:.1f}s to {slot_end:.1f}s)"):
                ads_for_slot = slot_ads.get(slot_number, [])
                
                if not ads_for_slot:
                    st.info(f"No ads assigned to this slot")
                    # Show recommended ads that could be assigned to this slot
                    recommended_for_slot = [
                        ad for ad in st.session_state.selected_ads 
                        if ad.get('recommended_slot') == slot_number
                    ]
                    if recommended_for_slot:
                        st.write("Recommended ads for this slot:")
                        for ad in recommended_for_slot:
                            st.write(f"- {ad['name']} ({ad['duration']}s)")
                            if st.button(f"Assign to Slot {slot_number}", key=f"assign_{slot_number}_{ad['id']}"):
                                ad['recommended_slot'] = slot_number
                                st.rerun()
                    continue
                
                # Calculate total duration of ads in this slot
                total_ad_duration = sum(ad['duration'] for ad in ads_for_slot)
                
                # Display warning if ads exceed slot duration
                if total_ad_duration > SLOT_DURATION:
                    st.error(f"⚠️ Total ad duration ({total_ad_duration}s) exceeds slot duration ({SLOT_DURATION}s)")
                else:
                    st.success(f"Total ad duration: {total_ad_duration}s / {SLOT_DURATION}s")
                
                # Display ads in this slot
                st.write("**Ads in this slot:**")
                for i, ad in enumerate(ads_for_slot):
                    col1, col2, col3 = st.columns([4, 1, 1])
                    with col1:
                        st.write(f"{i+1}. {ad['name']}")
                    with col2:
                        st.write(f"{ad['duration']}s")
                    with col3:
                        if st.button("❌", key=f"remove_{slot_number}_{ad['id']}"):
                            st.session_state.selected_ads.remove(ad)
                            st.rerun()
                
                # Calculate placements within slot
                current_time = slot_start
                for ad in ads_for_slot:
                    placement = {
                        'start': current_time,
                        'end': current_time + ad['duration'],
                        'duration': ad['duration'],
                        'ad_name': ad['name'],
                        'slot_number': slot_number,
                        'ad_id': ad['id']
                    }
                    st.session_state.ad_placements[ad['id']] = placement
                    current_time += ad['duration']
                
                # Visualize the slot timeline
                fig, ax = plt.subplots(figsize=(10, 1))
                ax.barh(0, SLOT_DURATION, left=slot_start, height=0.5, color='lightgray', alpha=0.3)
                
                current_time = slot_start
                for ad in ads_for_slot:
                    duration = ad['duration']
                    ax.barh(0, duration, left=current_time, height=0.5, color='#ff7f0e')
                    ax.text(current_time + duration/2, 0, 
                          ad['name'][:10] + "...", 
                          ha='center', va='center', color='white')
                    current_time += duration
                
                ax.set_xlim(slot_start, slot_end)
                ax.set_yticks([])
                ax.set_xlabel('Time (seconds)')
                st.pyplot(fig)
        
        # Full video timeline visualization
        st.subheader("Full Video Timeline")
        
        # Create segments list for timeline visualization
        video_duration = st.session_state.video_analysis['metadata']['duration']
        segments = []
        last_end = 0
        
        # Sort all placements by start time
        all_placements = sorted(
            st.session_state.ad_placements.values(),
            key=lambda x: x['start']
        )
        
        # Build segments list
        for placement in all_placements:
            # Add video segment before this ad
            if placement['start'] > last_end:
                segments.append({
                    'type': 'video',
                    'start': last_end,
                    'end': placement['start'],
                    'duration': placement['start'] - last_end
                })
            
            # Add ad segment
            segments.append({
                'type': 'ad',
                'ad': next(ad for ad in st.session_state.selected_ads if ad['id'] == placement['ad_id']),
                'start': placement['start'],
                'end': placement['end'],
                'duration': placement['duration'],
                'slot_number': placement['slot_number']
            })
            
            last_end = placement['end']
        
        # Add remaining video after last ad
        if last_end < video_duration:
            segments.append({
                'type': 'video',
                'start': last_end,
                'end': video_duration,
                'duration': video_duration - last_end
            })
        
        # Create full timeline visualization
        fig, ax = plt.subplots(figsize=(10, 2))
        
        for seg in segments:
            if seg['type'] == 'video':
                ax.barh(0, seg['duration'], left=seg['start'], height=0.5, color='#1f77b4')
            else:
                ax.barh(0, seg['duration'], left=seg['start'], height=0.5, color='#ff7f0e')
                label = f"Ad (S{seg['slot_number']})"
                ax.text(seg['start'] + seg['duration']/2, 0, 
                       label, ha='center', va='center', color='white')
        
        # Mark slot boundaries
        for slot in slots:
            slot_end = slot['start_time'] + SLOT_DURATION
            ax.axvspan(slot['start_time'], slot_end, color='yellow', alpha=0.1)
            ax.axvline(x=slot['start_time'], color='gray', linestyle='--', alpha=0.7)
            ax.axvline(x=slot_end, color='gray', linestyle='--', alpha=0.7)
            ax.text(slot['start_time'] + SLOT_DURATION/2, 0.7, 
                   f"Slot {slot['slot_number']}", ha='center', va='center')
        
        ax.set_yticks([])
        ax.set_xlabel('Time (seconds)')
        ax.grid(True, axis='x')
        st.pyplot(fig)
        
        # Segment preview controls
        st.subheader("Segment Preview")
        selected_segment = st.selectbox(
            "Select segment to preview",
            options=[f"{seg['type']} ({seg['start']:.1f}s-{seg['end']:.1f}s)" + 
                    (f" [Slot {seg['slot_number']}]" if 'slot_number' in seg else "")
                    for seg in segments],
            index=0
        )
        
        # Find selected segment
        seg_idx = [i for i, seg in enumerate(segments) 
                  if f"{seg['type']} ({seg['start']:.1f}s-{seg['end']:.1f}s)" + 
                  (f" [Slot {seg['slot_number']}]" if 'slot_number' in seg else "") == selected_segment][0]
        seg = segments[seg_idx]

        if seg['type'] == 'video':
            st.write(f"Video segment from {seg['start']:.1f}s to {seg['end']:.1f}s")
            try:
                cache_key = f"{seg['start']}_{seg['end']}"
                
                if cache_key in st.session_state.frame_cache:
                    preview_frame = st.session_state.frame_cache[cache_key]
                    st.image(preview_frame, caption=f"Video preview at {(seg['start'] + seg['end'])/2:.1f}s")
                else:
                    middle_time = (seg['start'] + seg['end']) / 2
                    cap = cv2.VideoCapture(st.session_state.video_analysis['metadata']['video_path'])
                    if cap.isOpened():
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frame_pos = int(middle_time * fps)
                        if cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos):
                            ret, frame = cap.read()
                            if ret and frame is not None:
                                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                st.session_state.frame_cache[cache_key] = frame
                                st.image(frame, caption=f"Video preview at {middle_time:.1f}s")
                        cap.release()
            except Exception as e:
                logger.error(f"Error previewing video segment: {str(e)}")
                st.error(f"Could not preview video segment: {str(e)}")
        else:
            st.write(f"Advertisement: {seg['ad']['name']}")
            st.write(f"Placed in Slot {seg['slot_number']} ({seg['start']:.1f}s-{seg['end']:.1f}s)")
            try:
                st.video(seg['ad']['path'])
            except:
                st.warning("Could not load ad preview")
        
        # Export options
        st.divider()
        st.subheader("Export Final Video")
        
        col1, col2 = st.columns(2)
        with col1:
            output_format = st.selectbox("Output Format", ["mp4", "mov", "mkv"])
            transitions = st.checkbox("Add Crossfade Transitions", True)
            mute_ads = st.checkbox("Mute Ads", False)
        with col2:
            quality = st.slider("Quality (1-10)", 1, 10, 6)
            st.write(f"Estimated size: {len(segments) * 5 + 10}MB")
        
        if st.button("Generate Final Video", type="primary"):
            with st.spinner("Generating video..."):
                with tempfile.TemporaryDirectory() as tmpdir:
                    try:
                        # Save main video
                        main_path = os.path.join(ASSETS_folder, tmpdir, "main_video.mp4")
                        shutil.copy(st.session_state.video_analysis['metadata']['video_path'], main_path)
                        
                        # Load main video clip
                        main_clip = VideoFileClip(main_path)
                        main_duration = main_clip.duration
                        main_width, main_height = main_clip.size
                        main_resolution = (main_width + main_width % 2, main_height + main_height % 2)
                        main_clip = main_clip.resize(newsize=main_resolution)

                        # Process ads
                        ad_info = []
                        for ad in st.session_state.selected_ads:
                            if ad['id'] not in st.session_state.ad_placements:
                                continue
                                
                            ad_path = os.path.join(ASSETS_folder, tmpdir, f"ad_{ad['id']}.mp4")
                            shutil.copy(ad['path'], ad_path)
                            ad_clip = VideoFileClip(ad_path).resize(newsize=main_resolution)
                            if mute_ads:
                                ad_clip = ad_clip.without_audio()
                            placement = st.session_state.ad_placements[ad['id']]
                            ad_info.append((placement['start'], ad_clip))

                        # Sort ads by placement time
                        ad_info.sort(key=lambda x: x[0])

                        # Merge video and ads
                        final_clips = []
                        last_cut = 0

                        for insert_time, ad_clip in ad_info:
                            if insert_time > main_duration:
                                continue
                            pre_clip = main_clip.subclip(last_cut, insert_time)
                            final_clips.append(pre_clip)
                            final_clips.append(ad_clip)
                            last_cut = insert_time

                        # Add remaining video
                        if last_cut < main_duration:
                            final_clips.append(main_clip.subclip(last_cut))

                        # Concatenate all clips
                        final_video = concatenate_videoclips(final_clips, method="compose")
                        output_path = os.path.join(ASSETS_folder, tmpdir, f"final_video.{output_format}")
                        final_video.write_videofile(
                            output_path, 
                            codec="libx264", 
                            audio_codec="aac",
                            bitrate=f"{quality*1000}k"
                        )

                        # Display success message
                        st.success("✅ Video successfully created!")
                        st.markdown(f"**Resolution:** {main_resolution[0]}x{main_resolution[1]}")
                        st.markdown(f"**Duration:** {final_video.duration:.2f} seconds")
                        
                        # Show final video
                        st.video(output_path)
                        
                        # Create download button
                        with open(output_path, 'rb') as f:
                            video_bytes = f.read()
                            st.download_button(
                                label="📥 Download Final Video",
                                data=video_bytes,
                                file_name=f"final_video.{output_format}",
                                mime=f"video/{output_format}"
                            )
                    
                    except Exception as e:
                        st.error(f"Video generation failed: {str(e)}")
                        logger.error(f"Video generation error: {str(e)}", exc_info=True)

def main():
    # Initialize session state and check environment
    init_session_state()
    check_environment()
    
    st.set_page_config(layout="wide")
    # Display GPU/CPU status
    if MODEL_DEVICE == "cuda":
        st.sidebar.success(f"GPU Detected: {torch.cuda.get_device_name(0)}")
    else:
        st.sidebar.warning("Running on CPU - GPU not detected")

     
    # Create necessary directories if they don't exist
    required_dirs = [
        "assets/sample_videos",
        "assets/ads",
        "assets/ads_json",
        "assets/processed_videos",
        "assets/processed_videos/frames"
    ]
    
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    mods_1 = st.sidebar.radio("Modules",["Smart Placement"])

    #################################################################
    # ------------- Recommendation System --------------------------
    #################################################################

    if mods_1 == "Smart Placement":
        mods = st.sidebar.radio("Select",["Smart Placement"])

        if mods == "Smart Placement":
            module = st.sidebar.radio("Module", [
                "Context2Content",
                # "AdGauge 360 - Predictive Creative Intelligence",
                "Contextual Media Matching",
                # "Synthetic Virtual Audience A/B Testing"
            ])
            
            # Module 1: SmartPlacement - Contextual Media Matching
            if module == "Context2Content":
                # st.sidebar.markdown("### Categories")
                smartplacement_tab, = st.tabs(["Context2Content Recommender"])

                with smartplacement_tab:
                    # st.sidebar.markdown("#### Ad Management Pages")
                    Context_Processing, Context2Content_Integration = st.tabs(["Context Processing", "Context2Content Integration"])
                    
                    with Context_Processing: # if ad_management_page == "Context Processing":
                        video_processing_page()
                    
                    with Context2Content_Integration: # elif ad_management_page == "Context2Content Integration":
                        ad_integration_page()
if __name__ == "__main__":
    main()

