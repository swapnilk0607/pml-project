import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from utils import save_uploaded_files, draw_bounding_boxes
from model_loader import run_detection_pipeline
from tabs.image_annotation.image_annotation import launch_annotation_tool

# Page Config
st.set_page_config(
    page_title="Cricket Object Detection",
    page_icon="🏏",
    layout="wide"
)

# Constants
INPUT_DIR = "input_images"

# Create input directory if it doesn't exist
Path(INPUT_DIR).mkdir(exist_ok=True)

# Initialize Session State
if 'results' not in st.session_state:
    st.session_state.results = None
if 'uploaded_count' not in st.session_state:
    st.session_state.uploaded_count = 0
if 'annotation_completed' not in st.session_state:
    st.session_state.annotation_completed = False
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Annotate Training Images"

def main():
    # Custom styling for header
    st.markdown("""
        <style>
        /* Header Animations */
        @keyframes gradient-shift {
            0% { background-position: 0% 50%; }
            50% { background-position: 100% 50%; }
            100% { background-position: 0% 50%; }
        }
        
        @keyframes rotate {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        
        @keyframes bounce {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        
        @keyframes slideInLeft {
            from { opacity: 0; transform: translateX(-50px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes slideInRight {
            from { opacity: 0; transform: translateX(50px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        @keyframes pulse {
            0%, 100% { transform: scale(1); }
            50% { transform: scale(1.05); }
        }
        
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0px); }
            50% { transform: translateY(-15px); }
        }
        
        @keyframes glow {
            0%, 100% { 
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2),
                            0 0 40px rgba(238, 119, 82, 0.3),
                            0 0 60px rgba(35, 166, 213, 0.2);
            }
            50% { 
                box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3),
                            0 0 60px rgba(238, 119, 82, 0.5),
                            0 0 80px rgba(35, 166, 213, 0.4);
            }
        }
        
        @keyframes sparkle {
            0%, 100% { opacity: 0; transform: scale(0); }
            50% { opacity: 1; transform: scale(1); }
        }
        
        @keyframes wave {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(100%); }
        }
        
        @keyframes colorShift {
            0% { filter: hue-rotate(0deg); }
            100% { filter: hue-rotate(360deg); }
        }
        
        @keyframes textGlow {
            0%, 100% { text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 0 0 20px rgba(255, 255, 255, 0.3); }
            50% { text-shadow: 3px 3px 6px rgba(0, 0, 0, 0.3), 0 0 30px rgba(255, 255, 255, 0.6), 0 0 40px rgba(255, 255, 255, 0.4); }
        }

        /* Cricket ball animation across header */
        @keyframes ball-move {
            0%   { left: -60px; top: 14%; }
            25%  { left: 25%;  top: 26%; }
            50%  { left: 55%;  top: 54%; }
            75%  { left: 80%;  top: 34%; }
            100% { left: calc(100% + 60px); top: 14%; }
        }
        
        /* Header Styling */
        .main-header {
            background: linear-gradient(-45deg, #ee7752, #e73c7e, #23a6d5, #23d5ab);
            background-size: 400% 400%;
            animation: gradient-shift 15s ease infinite, fadeIn 1s ease-out, glow 3s ease-in-out infinite;
            padding: 1.5rem 1.2rem;
            border-radius: 12px;
            margin-bottom: 1.2rem;
            text-align: center;
            position: relative;
            overflow: hidden;
        }
        
        .main-header::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: radial-gradient(circle, rgba(255,255,255,0.1) 0%, transparent 70%);
            animation: rotate 20s linear infinite;
            z-index: 0;
        }
        
        .main-header::after {
            content: '';
            position: absolute;
            top: 0;
            left: -100%;
            width: 100%;
            height: 100%;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.2), transparent);
            animation: wave 3s linear infinite;
            z-index: 1;
        }

        /* Animated cricket ball */
        .main-header .cricket-ball {
            position: absolute;
            width: 40px;
            height: 40px;
            top: 14%;
            left: -60px;
            z-index: 2; /* behind text (z=3), above bg */
            animation: ball-move 14s linear infinite;
            pointer-events: none;
        }

        .main-header .cricket-ball svg {
            width: 100%;
            height: 100%;
            filter: drop-shadow(0 6px 16px rgba(178,34,34,0.35));
            animation: rotate 2.5s linear infinite;
        }
        
        /* Sparkle effects */
        .sparkle {
            position: absolute;
            width: 4px;
            height: 4px;
            background: white;
            border-radius: 50%;
            z-index: 2;
            pointer-events: none;
        }
        
        .sparkle:nth-child(1) {
            top: 20%;
            left: 15%;
            animation: sparkle 2s ease-in-out infinite;
        }
        
        .sparkle:nth-child(2) {
            top: 60%;
            left: 80%;
            animation: sparkle 2.5s ease-in-out infinite 0.5s;
        }
        
        .sparkle:nth-child(3) {
            top: 40%;
            left: 40%;
            animation: sparkle 3s ease-in-out infinite 1s;
        }
        
        .sparkle:nth-child(4) {
            top: 70%;
            left: 25%;
            animation: sparkle 2.2s ease-in-out infinite 1.5s;
        }
        
        .sparkle:nth-child(5) {
            top: 30%;
            left: 90%;
            animation: sparkle 2.8s ease-in-out infinite 0.8s;
        }
        
        .main-header h1 {
            color: white;
            font-size: 1.8rem;
            font-weight: 800;
            margin-bottom: 0.4rem;
            letter-spacing: 0.5px;
            position: relative;
            z-index: 3;
            animation: fadeIn 1.2s ease-out, textGlow 2s ease-in-out infinite;
        }
        
        .main-header .emoji {
            display: inline-block;
            animation: bounce 2s ease-in-out infinite, float 3s ease-in-out infinite;
            font-size: 2rem;
            margin-right: 0.4rem;
            filter: drop-shadow(0 0 10px rgba(255, 255, 255, 0.5));
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.95);
            font-size: 0.9rem;
            font-weight: 300;
            margin: 0;
            position: relative;
            z-index: 3;
            text-shadow: 1px 1px 3px rgba(0, 0, 0, 0.2);
            animation: fadeIn 1.8s ease-out;
        }
        
        .main-header .tagline {
            display: inline-block;
            padding: 0.3rem 1rem;
            background: rgba(255, 255, 255, 0.15);
            border-radius: 50px;
            backdrop-filter: blur(10px);
            margin-top: 0.3rem;
            animation: fadeIn 1.5s ease-out, pulse 4s ease-in-out infinite;
            box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
            border: 1px solid rgba(255, 255, 255, 0.2);
            z-index: 3;
            position: relative;
        }
        
        /* Content Animations */
        .stMarkdown, .stHeader {
            animation: fadeIn 0.8s ease-out;
        }
        
        /* Button Animations */
        .stButton > button {
            transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            padding: 0.5rem 1rem;
            border-radius: 999px;
            background: linear-gradient(135deg, #667eea 0%, #23a6d5 100%);
            border: none;
            color: #fff;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(34, 139, 230, 0.25);
        }
        
        .stButton > button:hover {
            transform: translateY(-1px);
            box-shadow: 0 10px 24px rgba(34, 139, 230, 0.35);
            filter: brightness(1.05);
        }
        
        .stButton > button:active {
            transform: translateY(0);
        }
        
        .stButton > button::before {
            content: '';
            position: absolute;
            top: 50%;
            left: 50%;
            width: 0;
            height: 0;
            border-radius: 50%;
            background: rgba(255, 255, 255, 0.3);
            transform: translate(-50%, -50%);
            transition: width 0.6s, height 0.6s;
        }
        
        .stButton > button:hover::before {
            width: 300px;
            height: 300px;
        }
        
        /* Info/Warning/Success Box Animations */
        .stAlert {
            animation: slideInLeft 0.6s ease-out;
        }
        
        /* Image Animations */
        .stImage {
            animation: fadeIn 0.8s ease-out;
            transition: transform 0.3s ease;
        }
        
        .stImage:hover {
            transform: scale(1.02);
        }
        
        /* Expander Animations */
        .streamlit-expanderHeader {
            transition: all 0.3s ease;
        }
        
        .streamlit-expanderHeader:hover {
            background-color: rgba(151, 166, 195, 0.15);
            transform: translateX(5px);
        }
        
        /* Tabs: compact size + active state */
        .stTabs [data-baseweb="tab"] {
            transition: all 0.25s ease;
            padding: 6px 10px; /* smaller height */
            margin-right: 6px;
            border-radius: 10px;
            font-size: 0.88rem; /* smaller text */
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.10);
        }

        .stTabs [data-baseweb="tab"]:hover {
            transform: translateY(-1px);
            background: rgba(255,255,255,0.1);
        }

        /* Active/Selected tab */
        .stTabs [data-baseweb="tab"][aria-selected="true"] {
            /* Theme-adherent background: deep glass card */
            background: linear-gradient(135deg, rgba(26,34,48,0.95), rgba(26,34,48,0.82));
            border-color: rgba(102,126,234,0.8);
            color: #ffffff !important; /* override default red */
            box-shadow: 0 12px 26px rgba(34,139,230,0.28), inset 0 0 0 1px rgba(255,255,255,0.08);
            position: relative;
            transform: scale(1.04);
            font-weight: 800;
            letter-spacing: 0.2px;
        }

        /* Inactive tabs */
        .stTabs [data-baseweb="tab"][aria-selected="false"] {
            opacity: 0.65;
            filter: saturate(0.8);
        }

        /* Active indicator bar */
        .stTabs [data-baseweb="tab"][aria-selected="true"]::after {
            content: '';
            position: absolute;
            left: 4px; right: 4px; bottom: -4px;
            height: 4px;
            border-radius: 4px;
            background: linear-gradient(90deg, #667eea, #23a6d5);
            box-shadow: 0 3px 10px rgba(34,139,230,0.45);
        }

        /* Active left accent bar for extra distinction */
        .stTabs [data-baseweb="tab"][aria-selected="true"]::before {
            content: '';
            position: absolute;
            left: -4px; top: 5px; bottom: 5px;
            width: 4px;
            border-radius: 4px;
            background: linear-gradient(180deg, #23a6d5, #23d5ab);
            box-shadow: 0 0 10px rgba(35,166,213,0.55);
        }

        /* Remove default focus red/outline */
        .stTabs [data-baseweb="tab"]:focus {
            outline: none;
            color: #eaeefb;
        }
        
        /* Metric Card Animations */
        .css-1r6slb0 {
            animation: fadeIn 0.8s ease-out;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }
        
        .css-1r6slb0:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
        }
        
        /* File Uploader Animation */
        .stFileUploader {
            animation: slideInRight 0.6s ease-out;
        }
        
        /* Dataframe Animation */
        .stDataFrame {
            animation: fadeIn 1s ease-out;
        }
        
        /* Spinner Custom Style */
        .stSpinner > div {
            border-color: #667eea;
            border-right-color: transparent;
        }
        
        /* Sidebar Animation */
        [data-testid="stSidebar"] {
            animation: slideInLeft 0.5s ease-out;
            background: linear-gradient(180deg, rgba(20,25,35,0.85) 0%, rgba(20,25,35,0.95) 100%);
            backdrop-filter: blur(12px);
            box-shadow: 0 0 30px rgba(0,0,0,0.35);
        }

        /* Sidebar container card */
        [data-testid="stSidebar"] .css-1d391kg, 
        [data-testid="stSidebar"] .css-1v3fvcr {
            border-radius: 18px;
            padding: 1rem;
            background: linear-gradient(135deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            border: 1px solid rgba(255,255,255,0.12);
        }

        /* Stylish radio navigation */
        /* Sidebar buttons styling */
        [data-testid="stSidebar"] .stButton > button {
            width: 100%;
            margin: 6px 0;
            border-radius: 12px;
            background: linear-gradient(135deg, rgba(102,126,234,0.85), rgba(35,166,213,0.85));
            border: 1px solid rgba(255,255,255,0.15);
            color: #fff;
            font-weight: 600;
            box-shadow: 0 6px 18px rgba(34, 139, 230, 0.25);
        }

        [data-testid="stSidebar"] .stButton > button:hover {
            transform: translateX(2px);
            box-shadow: 0 10px 24px rgba(34, 139, 230, 0.35);
            filter: brightness(1.06);
        }

        /* Logo area tweaks */
        [data-testid="stSidebar"] svg {
            filter: drop-shadow(0 2px 8px rgba(35,166,213,0.35));
        }

        /* Sidebar header/title (optional) */
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
            color: #ffffff;
            text-shadow: 1px 1px 3px rgba(0,0,0,0.25);
            animation: fadeIn 0.8s ease-out;
        }
        
        /* Radio Button Hover Effect */
        .stRadio > label {
            transition: all 0.2s ease;
        }
        
        .stRadio > label:hover {
            transform: translateX(5px);
            color: #667eea;
        }
        </style>
        <div class="main-header">
            <span class="sparkle"></span>
            <span class="sparkle"></span>
            <span class="sparkle"></span>
            <span class="sparkle"></span>
            <span class="sparkle"></span>
            <div class="cricket-ball">
                <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg" aria-hidden="true">
                    <circle cx="12" cy="12" r="10" fill="#ff6b6b" stroke="#b22222" stroke-width="1.2"/>
                    <path d="M4 12 H20" stroke="#ffffff" stroke-width="1" opacity="0.9"/>
                    <path d="M4 14 H20" stroke="#ffffff" stroke-width="0.8" opacity="0.75"/>
                    <path d="M4 10 H20" stroke="#ffffff" stroke-width="0.8" opacity="0.6"/>
                </svg>
            </div>
            <h1><span class="emoji">🏏</span>Cricket Object Detection System</h1>
            <div class="tagline">
                <p>AI/ML-Powered Detection for Bats, Balls & Stumps</p>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Sidebar: Logo + Navigation (buttons)
    st.sidebar.markdown(
            """
            <style>
            @keyframes logoGlow { 0%,100%{filter: drop-shadow(0 0 6px rgba(35,166,213,0.45));} 50%{filter: drop-shadow(0 0 12px rgba(35,166,213,0.85));} }
            @keyframes shine { 0%{transform: translateX(-120%);} 100%{transform: translateX(120%);} }
            .brand-wrap { display:flex; align-items:center; gap:12px; padding:12px 8px 18px 8px; }
            .brand-text { display:flex; flex-direction:column; }
            .brand-title { font-weight:800; font-size:1.6rem; letter-spacing:.8px; color:#ffffff; text-shadow:1px 1px 3px rgba(0,0,0,.25); line-height:1.05; }
            .brand-sub { font-size:.82rem; color:#cfe8ff; }
            .logo { width:60px; height:60px; animation: logoGlow 4s ease-in-out infinite; position:relative; }
            .logo .shine { position:absolute; inset:0; background: linear-gradient(90deg, transparent, rgba(255,255,255,.28), transparent); animation: shine 5s linear infinite; mix-blend-mode: overlay; border-radius:50%; }
            </style>
            <div class="brand-wrap">
                <div class="logo">
                    <svg viewBox="0 0 64 64" xmlns="http://www.w3.org/2000/svg">
                        <defs>
                            <linearGradient id="ring" x1="0" x2="1" y1="0" y2="1">
                                <stop offset="0%" stop-color="#23a6d5"/>
                                <stop offset="100%" stop-color="#23d5ab"/>
                            </linearGradient>
                            <radialGradient id="ball" cx="50%" cy="50%" r="50%">
                                <stop offset="0%" stop-color="#ffd1dc"/>
                                <stop offset="100%" stop-color="#ff6b6b"/>
                            </radialGradient>
                        </defs>
                        <!-- Outer detection ring -->
                        <circle cx="32" cy="32" r="28" fill="none" stroke="url(#ring)" stroke-width="3.5"/>
                        <!-- Camera focus corners -->
                        <path d="M14 22 h4 a2 2 0 0 1 2 2 v4" stroke="#23a6d5" stroke-width="2" fill="none" stroke-linecap="round"/>
                        <path d="M50 22 h-4 a2 2 0 0 0 -2 2 v4" stroke="#23a6d5" stroke-width="2" fill="none" stroke-linecap="round"/>
                        <path d="M14 42 h4 a2 2 0 0 0 2 -2 v-4" stroke="#23d5ab" stroke-width="2" fill="none" stroke-linecap="round"/>
                        <path d="M50 42 h-4 a2 2 0 0 1 -2 -2 v-4" stroke="#23d5ab" stroke-width="2" fill="none" stroke-linecap="round"/>
                        <!-- Stumps -->
                        <rect x="28" y="34" width="2" height="10" rx="1" fill="#cfd7ff"/>
                        <rect x="32" y="33" width="2" height="11" rx="1" fill="#cfd7ff"/>
                        <rect x="36" y="34" width="2" height="10" rx="1" fill="#cfd7ff"/>
                        <!-- Bat (tilted) -->
                        <path d="M20 40 l10 -10 c1 -1 2 -1 3 0 l1 1 c1 1 1 2 0 3 l-10 10 c-1 1 -2 1 -3 0 l-1 -1 c-1 -1 -1 -2 0 -3 z" fill="#f5c27b" stroke="#e2a85a" stroke-width="1"/>
                        <!-- Ball -->
                        <circle cx="42" cy="24" r="5" fill="url(#ball)" stroke="#b22222" stroke-width="1"/>
                        <!-- Seam lines on ball -->
                        <path d="M38 24 h8" stroke="#ffffff" stroke-width="0.8" opacity="0.8"/>
                        <path d="M38 25.5 h8" stroke="#ffffff" stroke-width="0.6" opacity="0.7"/>
                    </svg>
                    <div class="shine"></div>
                </div>
                <div class="brand-text">
                    <div class="brand-title">Cricket Vision</div>
                    <div class="brand-sub">Vision AI Toolkit</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
    )

    # Navigation buttons (Annotate first, then Feature Extraction)
    nav_annotate = st.sidebar.button("🏷️ Annotate Training Images")
    nav_feature = st.sidebar.button("🧩 ML Model Creation")
    nav_test_model = st.sidebar.button("🧪 Test Model")

    if nav_annotate:
        st.session_state.current_page = "Annotate Training Images"
    elif nav_feature:
        st.session_state.current_page = "Feature Extraction"
    elif nav_test_model:
        st.session_state.current_page = "Test Model"
    

    # Sidebar footer: Team card pinned to bottom with neon glow effects (always render)
    st.sidebar.markdown(
            """
            <style>
            /* Make sidebar a flex column to push footer to bottom */
            [data-testid="stSidebar"] > div:first-child { display: flex; flex-direction: column; height: 100%; }
            [data-testid="stSidebar"] .sidebar-footer { margin-top: auto; }
            .team-card {
                    border: 1px solid rgba(0, 255, 255, 0.35);
                    background: radial-gradient(120% 120% at 0% 0%, rgba(10, 20, 40, 0.95) 20%, rgba(10, 20, 40, 0.8) 60%),
                                            linear-gradient(180deg, rgba(0,255,255,0.08), rgba(255,0,255,0.06));
                    border-radius: 14px; padding: 12px 12px 10px 12px; color: #eaf2ff;
                    position: relative; overflow: hidden;
                    box-shadow: 0 0 14px rgba(0, 255, 255, 0.25), inset 0 0 12px rgba(0, 255, 255, 0.12);
            }
            .team-card::before {
                    content: ""; position: absolute; inset: -2px; border-radius: 16px;
                    background: conic-gradient(from 180deg at 50% 50%, rgba(0,255,255,0.0), rgba(0,255,255,0.6), rgba(255,0,255,0.5), rgba(0,255,255,0.0));
                    filter: blur(14px); opacity: 0.35; pointer-events: none;
                    animation: rotateGlow 10s linear infinite;
            }
            @keyframes rotateGlow { 0%{ transform: rotate(0deg) } 100%{ transform: rotate(360deg) } }
            .team-title { font-weight: 800; font-size: 1.02rem; letter-spacing: .6px; margin-bottom: 6px; display:flex; align-items:center; gap:10px; text-shadow: 0 0 6px rgba(0,255,255,0.35); }
            .team-title .emblem { width: 18px; height: 18px; border-radius: 50%; background: radial-gradient(circle, #00e7ff 0%, #ff00ff 70%); box-shadow: 0 0 10px rgba(0,255,255,0.65), 0 0 18px rgba(255,0,255,0.35); }
            .team-list { list-style: none; padding: 0; margin: 6px 0 0 0; display: grid; gap: 6px; }
            .team-list li { display:flex; align-items:center; gap:10px; font-size: .93rem; color: #e6f9ff; text-shadow: 0 0 6px rgba(0,255,255,0.25); }
            .pulse-dot { width: 8px; height: 8px; border-radius: 50%; background: #00e7ff; box-shadow: 0 0 8px rgba(0,231,255,0.9), 0 0 16px rgba(0,231,255,0.45); animation: pulse 1.8s ease-out infinite; }
            @keyframes pulse { 0%{transform: scale(0.9); box-shadow: 0 0 8px rgba(0,231,255,0.9), 0 0 0 0 rgba(0,231,255,0.0)} 60%{transform: scale(1); box-shadow: 0 0 14px rgba(0,231,255,0.95), 0 0 14px rgba(0,231,255,0.32)} 100%{transform: scale(0.9); box-shadow: 0 0 8px rgba(0,231,255,0.9), 0 0 0 0 rgba(0,231,255,0.0)} }
            .team-card:hover { box-shadow: 0 0 18px rgba(0,255,255,0.45), inset 0 0 16px rgba(0,255,255,0.18); }
            </style>
            <div class="sidebar-footer">
                <div class="team-card">
                    <div class="team-title"><span class="emblem"></span>Group -3 </div>
                    <ul class="team-list">
                        <li><span class="pulse-dot"></span> Abhishek Jaiswal</li>
                        <li><span class="pulse-dot"></span> Siddhesh Shirke</li>
                        <li><span class="pulse-dot"></span> Sushant Kambli</li>
                        <li><span class="pulse-dot"></span> Swapnil Katale</li>
                    </ul>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
    )

    page = st.session_state.current_page

    if page == "Annotate Training Images":
        render_annotation_section()
    elif page == "Feature Extraction":
        render_feature_extraction_section()
    elif page == "Test Model":
        render_test_model_section()


def render_annotation_section():
    # st.header("🏷️ Annotate Training Images")
    # st.markdown("Click the button below to launch the annotation tool and tag your training images.")
    
    # Create required directories
    annotation_dir = "training_images"
    Path(annotation_dir).mkdir(exist_ok=True)
    
    # Button to launch annotation tool (compact, centered within a container)
    st.markdown("")
    with st.container(border=True):
        st.markdown("### Start Annotation")
        st.caption("Launch the annotation tool to tag training images.")
        btn_cols = st.columns([2, 1, 2])
        with btn_cols[1]:
            annotate_clicked = st.button("✏️ Annotate Image", type="primary", use_container_width=False)
    if annotate_clicked:
        try:
            with st.spinner("Launching annotation tool..."):
                # Launch annotation tool with training images
                launch_annotation_tool(
                    image_folder=annotation_dir,
                    output_csv="tabs/image_annotation/output/training_annotations.csv",
                    processed_folder="tabs/image_annotation/output/annotated_training_images"
                )
            st.session_state.annotation_completed = True
            st.success("✅ Annotation tool closed. Displaying annotated images below.")
            st.rerun()
        except Exception as e:
            st.error(f"❌ Error launching annotation tool: {e}")
    
    # Display annotated images only after annotation tool has been executed
    if st.session_state.annotation_completed:
        annotated_dir = "tabs/image_annotation/output/annotated_training_images"
        if os.path.exists(annotated_dir):
            annotated_images = [f for f in os.listdir(annotated_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            if annotated_images:
                st.markdown("---")
                st.subheader("📋 Annotated Training Images")
                st.info(f"✅ {len(annotated_images)} annotated image(s) available")
                
                # Display annotated images in grid (2 columns for larger images)
                cols = st.columns(2)
                for idx, img_name in enumerate(annotated_images):
                    col = cols[idx % 2]
                    img_path = os.path.join(annotated_dir, img_name)
                    
                    with col:
                        st.image(img_path, caption=f"Annotated: {img_name}", use_container_width=True)

def render_feature_extraction_section():
    import os
    import cv2
    # ML model creation: gather images and run extraction
    st.header("🧩 ML Model Creation")
    st.caption("Extract features from images and train the ML model using the generated dataset.")

    # Tabs within Feature Extraction
    tab_extract, tab_train = st.tabs(["Feature Extraction", "Train Model"])

    with tab_extract:
        # Define input folder under feature_extraction
        input_folder = "tabs/feature_extraction/input"
        os.makedirs(input_folder, exist_ok=True)

        # List image files
        valid_exts = (".png", ".jpg", ".jpeg", ".webp")
        image_paths = [
            os.path.join(input_folder, f)
            for f in os.listdir(input_folder)
            if f.lower().endswith(valid_exts)
        ]

        col_a, col_b = st.columns([2,1])
        with col_a:
            st.write(f"Images found: {len(image_paths)}")
            if not image_paths:
                st.warning("No images found. Add images to the input folder to proceed.")
            else:
                st.markdown("#### Input Preview")
                # Simple, attractive gallery: 3 columns, hover lift
                gallery_cols = st.columns(3)
                max_show = min(len(image_paths), 12)
                for idx, p in enumerate(image_paths[:max_show]):
                    with gallery_cols[idx % 3]:
                        st.image(p, caption=os.path.basename(p), use_container_width=True)

                # Subtle hover style for images
                st.markdown(
                    """
                    <style>
                    .stImage img { transition: transform .2s ease, box-shadow .2s ease; border-radius: 10px; }
                    .stImage img:hover { transform: translateY(-2px); box-shadow: 0 10px 18px rgba(0,0,0,.25); }
                    </style>
                    """,
                    unsafe_allow_html=True,
                )

        with col_b:
            run_btn = st.button("⚙️ Run Feature Extraction", type="primary")

        if run_btn and image_paths:
            try:
                # Import class to allow per-image progress updates
                from tabs.feature_extraction.feature_extraction import EnhancedCricketFeatureExtractor
                output_csv = "tabs/feature_extraction/output/input_images_Feature.csv"

                progress = st.progress(0)
                status = st.empty()
                table_placeholder = st.empty()
                total = len(image_paths)

                # Group by directory and process with progress
                grouped = {}
                for p in image_paths:
                    dirpath = os.path.dirname(p) or "."
                    grouped.setdefault(dirpath, []).append(os.path.basename(p))

                dfs = []
                status_rows = []
                processed = 0
                for dirpath, files in grouped.items():
                    extractor = EnhancedCricketFeatureExtractor(data_dir=dirpath)
                    for f in files:
                        status.info(f"Processing: {os.path.join(dirpath, f)}")
                        extractor.extract_features_from_image(f)
                        processed += 1
                        progress.progress(int(processed * 100 / total))
                        # Update status table
                        status_rows.append({"file": os.path.join(dirpath, f), "status": "processed"})
                        import pandas as pd
                        df_status = pd.DataFrame(status_rows)
                        table_placeholder.table(df_status)
                        # Attractive table styling
                        st.markdown(
                            """
                            <style>
                            /* Style the status table for a sleeker look */
                            .stTable, .stDataFrame table {
                                border-collapse: separate !important;
                                border-spacing: 0 !important;
                                border-radius: 12px !important;
                                overflow: hidden !important;
                                box-shadow: 0 10px 24px rgba(0,0,0,0.25);
                            }
                            .stTable thead tr, .stDataFrame thead tr {
                                background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(35,166,213,0.25)) !important;
                                color: #eaf2ff !important;
                            }
                            .stTable tbody tr:nth-child(odd), .stDataFrame tbody tr:nth-child(odd) {
                                background-color: rgba(255,255,255,0.03) !important;
                            }
                            .stTable tbody tr:hover, .stDataFrame tbody tr:hover {
                                background-color: rgba(255,255,255,0.06) !important;
                            }
                            .stTable th, .stTable td, .stDataFrame th, .stDataFrame td {
                                padding: 10px 14px !important;
                                border-bottom: 1px solid rgba(255,255,255,0.06) !important;
                            }
                            </style>
                            """,
                            unsafe_allow_html=True,
                        )

                    # Write interim to temp and collect
                    temp_csv = "tabs/feature_extraction/output/temp_feature_chunk.csv"
                    df_chunk = extractor.write_features_to_file(filename=temp_csv)
                    dfs.append(df_chunk)

                # Concatenate and write final CSV
                import pandas as pd
                if dfs:
                    df = pd.concat(dfs, ignore_index=True)
                    import os
                    os.makedirs(os.path.dirname(output_csv) or '.', exist_ok=True)
                    df.to_csv(output_csv, index=False)
                else:
                    df = pd.DataFrame()

                progress.progress(100)
                status.empty()
                st.success(f"✅ Features extracted. Saved to {output_csv}")
                if not df.empty:
                    # Animated container for the feature preview
                    st.markdown(
                        """
                        <style>
                        @keyframes pulseGlow {
                            0% { box-shadow: 0 0 0 rgba(46, 134, 193, 0.0); }
                            50% { box-shadow: 0 0 22px rgba(46, 134, 193, 0.35); }
                            100% { box-shadow: 0 0 0 rgba(46, 134, 193, 0.0); }
                        }
                        .feature-preview-card {
                            border-radius: 14px;
                            padding: 12px 12px 4px 12px;
                            background: linear-gradient(180deg, rgba(255,255,255,0.04), rgba(255,255,255,0.02));
                            border: 1px solid rgba(255,255,255,0.08);
                            animation: pulseGlow 3s ease-in-out infinite;
                        }
                        /* DataFrame grid styling */
                        .stDataFrame table {
                            border-collapse: separate !important;
                            border-spacing: 0 !important;
                            border-radius: 12px !important;
                            overflow: hidden !important;
                            box-shadow: 0 10px 24px rgba(0,0,0,0.25);
                        }
                        .stDataFrame thead tr {
                            background: linear-gradient(135deg, rgba(102,126,234,0.25), rgba(35,166,213,0.25)) !important;
                            color: #eaf2ff !important;
                        }
                        .stDataFrame tbody tr:nth-child(odd) {
                            background-color: rgba(255,255,255,0.03) !important;
                        }
                        .stDataFrame tbody tr:hover {
                            background-color: rgba(255,255,255,0.06) !important;
                            transition: background-color 180ms ease-in-out;
                        }
                        .stDataFrame th, .stDataFrame td {
                            padding: 9px 12px !important;
                            border-bottom: 1px solid rgba(255,255,255,0.06) !important;
                        }
                        .stDataFrame tbody tr td:first-child {
                            color: rgba(255,255,255,0.65) !important;
                        }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown('<div class="feature-preview-card">', unsafe_allow_html=True)
                    # Header chips with quick stats
                    total_rows = len(df)
                    unique_images = df['image'].nunique() if 'image' in df.columns else total_rows
                    # Note: avoid f-string for CSS due to braces. Split into two blocks.
                    st.markdown(
                        """
                        <style>
                        .chip-row { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
                        .chip { border-radius: 999px; padding: 6px 12px; font-size: 0.85rem; 
                                border: 1px solid rgba(255,255,255,0.12); 
                                background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
                                color: #dfe8ff; }
                        .chip .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
                        .chip.hog .dot { background:#66a6ea; }
                        .chip.lbp .dot { background:#23a6d5; }
                        .chip.color .dot { background:#f7b733; }
                        .chip.edge .dot { background:#9b59b6; }
                        .pulse-btn { animation: pulseSoft 2.2s ease-in-out infinite; }
                        @keyframes pulseSoft { 0% { box-shadow: 0 0 0 rgba(102,166,234,0.0);} 50% { box-shadow:0 0 18px rgba(102,166,234,0.35);} 100% { box-shadow:0 0 0 rgba(102,166,234,0.0);} }
                        </style>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f"""
                        <div class=\"chip-row\"> 
                            <div class=\"chip\"><span class=\"dot\" style=\"background:#2ecc71\"></span>{unique_images} images</div>
                            <div class=\"chip\"><span class=\"dot\" style=\"background:#e74c3c\"></span>{total_rows} rows</div>
                            <div class=\"chip hog\"><span class=\"dot\"></span>HOG features</div>
                            <div class=\"chip lbp\"><span class=\"dot\"></span>LBP features</div>
                            <div class=\"chip color\"><span class=\"dot\"></span>Color stats</div>
                            <div class=\"chip edge\"><span class=\"dot\"></span>Edge/shape</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                    st.dataframe(df.head(50), use_container_width=True)
                    st.markdown('</div>', unsafe_allow_html=True)
                    # Provide a direct download button
                    csv_data = df.to_csv(index=False).encode('utf-8')
                    # Animated download button wrapper
                    st.markdown('<div class="pulse-btn" style="display:inline-block;border-radius:10px;">', unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download Feature CSV",
                        data=csv_data,
                        file_name="input_images_Feature.csv",
                        mime="text/csv"
                    )
                    st.markdown('</div>', unsafe_allow_html=True)
                else:
                    st.info("No feature rows generated. Check input images and try again.")
            except Exception as e:
                st.error(f"❌ Feature extraction failed: {e}")

    with tab_train:
        st.markdown("### Train Model")
        st.caption("Train the ML model using the extracted features CSV.")
        # Training UI: button + animated feedback
        # Pre-flight: verify features CSV exists
        features_csv = "tabs/model_creation/input/enhanced_features_consolidation.csv"
        import os
        if not os.path.exists(features_csv):
            st.warning("The features CSV was not found. Please run feature extraction first to generate 'enhanced_features_consolidation.csv'.")
            st.caption("Expected path: tabs/model_creation/input/enhanced_features_consolidation.csv")
        train_btn = st.button("🚀 Train the ML Model", type="primary", disabled=not os.path.exists(features_csv))
        if train_btn:
            try:
                # Animated training banner
                st.markdown(
                    """
                    <style>
                    @keyframes trainPulse { 0%{box-shadow:0 0 0 rgba(35,166,213,0.0)} 50%{box-shadow:0 0 22px rgba(35,166,213,0.35)} 100%{box-shadow:0 0 0 rgba(35,166,213,0.0)} }
                    .train-card { border:1px solid rgba(255,255,255,0.12); border-radius:14px; padding:12px; background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03)); animation: trainPulse 2.6s ease-in-out infinite; }
                    </style>
                    <div class="train-card">🛠️ Initializing training… Preparing data and models.</div>
                    """,
                    unsafe_allow_html=True,
                )
                with st.spinner("Training in progress… This may take a moment."):
                    # Import wrapper and run
                    from tabs.model_creation.model_creation import run_full_pipeline
                    _results = run_full_pipeline(features_csv, use_pca=True, n_features=250)
                st.success("✅ Training completed successfully.")
                st.balloons()
                # Summary chips
                try:
                    if isinstance(_results, dict) and len(_results) > 0:
                        best_name = max(_results, key=lambda m: _results[m].get('cv_mean', 0))
                        best_score = _results[best_name].get('cv_mean', 0)
                        cols_chips = st.columns(3)
                        with cols_chips[0]:
                            st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(35,166,213,0.15);border:1px solid rgba(35,166,213,0.35);'>🏆 Best Model: <strong>{best_name}</strong></div>", unsafe_allow_html=True)
                        with cols_chips[1]:
                            st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(46,204,113,0.15);border:1px solid rgba(46,204,113,0.35);'>📈 Mean CV F1: <strong>{best_score:.4f}</strong></div>", unsafe_allow_html=True)
                        with cols_chips[2]:
                            models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
                            st.markdown(f"<div style='display:inline-block;padding:6px 10px;border-radius:999px;background:rgba(241,196,15,0.15);border:1px solid rgba(241,196,15,0.35);'>💾 Saved To: <strong>{models_dir}</strong></div>", unsafe_allow_html=True)
                except Exception:
                    pass
                # Display saved evaluation visuals if available
                models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'models'))
                cm_path = os.path.join(models_dir, 'confusion_matrix.png')
                cr_path = os.path.join(models_dir, 'classification_report.png')
                if os.path.exists(cm_path) or os.path.exists(cr_path):
                    st.markdown("#### Evaluation Visuals")
                    cols = st.columns(2)
                    if os.path.exists(cm_path):
                        with cols[0]:
                            st.image(cm_path, caption="Confusion Matrix", use_container_width=True)
                    if os.path.exists(cr_path):
                        with cols[1]:
                            st.image(cr_path, caption="Classification Report", use_container_width=True)
            except Exception as e:
                st.error(f"❌ Training failed: {e}")

def render_test_model_section():
    st.header("🧪 Test Model")
    st.caption("Upload a single image to run object classification.")

    # Page styling for an attractive test area
    st.markdown(
        """
        <style>
        .test-card { 
            border: 1px solid rgba(255,255,255,0.12); 
            border-radius: 16px; 
            padding: 18px; 
            background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
            box-shadow: 0 10px 26px rgba(0,0,0,0.25);
            animation: fadeIn 0.6s ease-out;
        }
        .test-side { 
            border-radius: 12px; 
            padding: 12px; 
            background: linear-gradient(135deg, rgba(35,166,213,0.18), rgba(102,126,234,0.16));
            border: 1px solid rgba(35,166,213,0.28);
            color: #eaf2ff;
        }
        .preview-card {
            border-radius: 14px; padding: 10px; 
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.10);
        }
        .output-card {
            border-radius: 14px; padding: 12px;
            background: linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02));
            border: 1px solid rgba(255,255,255,0.10);
            min-height: 280px; position: relative;
        }
        /* Loader + overlay styles */
        .overlay {
            position: relative;
            border-radius: 12px;
            border: 1px dashed rgba(35,166,213,0.35);
            background: radial-gradient(circle at 30% 20%, rgba(35,166,213,0.10), rgba(102,126,234,0.08));
            padding: 16px; text-align: center; color: #eaf2ff;
        }
        .loader { width: 54px; height: 54px; border: 4px solid rgba(255,255,255,0.16); border-top-color: #23a6d5; border-radius: 50%; margin: 12px auto; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .scanline { position: relative; height: 6px; border-radius: 999px; overflow: hidden; background: rgba(255,255,255,0.10); margin: 12px 0; }
        .scanline::after {
            content: ""; position: absolute; left: -40%; top: 0; height: 100%; width: 40%;
            background: linear-gradient(90deg, rgba(35,166,213,0.0), rgba(35,166,213,0.6), rgba(35,166,213,0.0));
            animation: sweep 1.8s ease-in-out infinite;
        }
        @keyframes sweep { 0%{ left:-40% } 50%{ left:80% } 100%{ left:-40% } }
        .success-banner { border: 1px solid rgba(46, 204, 113, 0.35); background: linear-gradient(180deg, rgba(46, 204, 113, 0.18), rgba(46, 204, 113, 0.08)); padding: 12px; border-radius: 12px; color: #eaf2ff; }
        .success-check { display:inline-block; width: 18px; height: 18px; border-radius: 50%; background: #2ecc71; box-shadow: 0 0 0 0 rgba(46, 204, 113, 0.6); animation: pop 1.2s ease-out forwards; margin-right:8px; vertical-align: -3px; }
        @keyframes pop { 0%{ transform: scale(0.3); box-shadow: 0 0 0 0 rgba(46,204,113,0.6) } 50%{ transform: scale(1.05); box-shadow: 0 0 0 12px rgba(46,204,113,0.0) } 100%{ transform: scale(1); box-shadow: 0 0 0 0 rgba(46,204,113,0.0) } }
        @keyframes shimmer {
            0% { background-position: -1000px 0; }
            100% { background-position: 1000px 0; }
        }
        .skeleton {
            height: 220px; border-radius: 12px;
            background: linear-gradient(90deg, rgba(255,255,255,0.06) 25%, rgba(255,255,255,0.10) 50%, rgba(255,255,255,0.06) 75%);
            background-size: 1000px 100%; animation: shimmer 1.8s linear infinite;
        }
        .badge {
            display:inline-block; padding:6px 10px; border-radius:999px;
            background: rgba(35,166,213,0.15); border:1px solid rgba(35,166,213,0.35);
            font-size: 0.85rem; color:#dfe8ff;
        }
        .hint { color: #dfe8ff; opacity: 0.85; }
        .chip-row { display:flex; gap:10px; flex-wrap:wrap; margin: 6px 0 12px; }
        .chip { border-radius: 999px; padding: 6px 12px; font-size: 0.85rem; 
                border: 1px solid rgba(255,255,255,0.12); 
                background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
                color: #dfe8ff; }
        .chip .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:8px; }
        .chip.format .dot { background:#23a6d5; }
        .chip.size .dot { background:#667eea; }
        .chip.tip .dot { background:#2ecc71; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Session state flags
    if 'show_test_uploader' not in st.session_state:
        st.session_state.show_test_uploader = False
    if 'test_model_image' not in st.session_state:
        st.session_state.test_model_image = None
    # Persist last output image to avoid duplicate layouts on rerun
    if 'test_model_output_path' not in st.session_state:
        st.session_state.test_model_output_path = None

    with st.container():
        # st.markdown("#### Upload a test image")
        # st.markdown('<div class="test-card">', unsafe_allow_html=True)
        # Single CTA button to reveal uploader
        reveal = st.button("📤 Upload Test Image", type="primary")
        if reveal:
            st.session_state.show_test_uploader = True

        # Place uploader strictly below the button using a placeholder
        uploader_ph = st.empty()
        # Show uploader only until an image is chosen
        if st.session_state.show_test_uploader and st.session_state.test_model_image is None:
            test_image = uploader_ph.file_uploader(
                "",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=False,
                key="test_model_uploader",
                label_visibility="collapsed",
            )
            if test_image is not None:
                st.session_state.test_model_image = test_image
                # Hide uploader after selection
                st.session_state.show_test_uploader = False
        else:
            uploader_ph.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.test_model_image is not None:
        st.success("✅ Image selected")
        cols_io = st.columns([1, 1])
        with cols_io[0]:
            st.markdown("#### Input Image")
            # st.markdown('<div class="preview-card">', unsafe_allow_html=True)
            st.image(st.session_state.test_model_image, caption=None, width=420)
            st.markdown('</div>', unsafe_allow_html=True)
            run_clicked = st.button("🔍 Run Classification", type="primary")
            st.markdown('<span class="hint">The model will return predicted class and confidence.</span>', unsafe_allow_html=True)
        with cols_io[1]:
            st.markdown("#### Classified Image by ML Model")
            output_placeholder = st.empty()
            if st.session_state.test_model_output_path:
                output_placeholder.image(st.session_state.test_model_output_path, caption=None, width=420)
            else:
                output_placeholder.markdown('<span class="badge">Awaiting classification</span><div class="skeleton"></div>', unsafe_allow_html=True)

        if run_clicked:
            import time, os
            # Animated overlay inside the output area
            loader_html = """
            <div class='overlay'>
              <div class='loader'></div>
              <div>🔎 Running classification…</div>
              <div class='scanline'></div>
            </div>
            """
            output_placeholder.markdown(loader_html, unsafe_allow_html=True)

            progress = st.progress(0)
            status = st.empty()
            steps = [
                ("Loading model", 20),
                ("Preprocessing image", 40),
                ("Inference", 80),
                ("Postprocessing", 95),
                ("Preparing output", 100),
            ]
            for label, pct in steps:
                status.info(f"{label}…")
                progress.progress(pct)
                time.sleep(0.6)

            output_dir = "tabs/test_model/output"
            os.makedirs(output_dir, exist_ok=True)
            valid_exts = (".png", ".jpg", ".jpeg", ".webp")
            files = [
                os.path.join(output_dir, f)
                for f in os.listdir(output_dir)
                if f.lower().endswith(valid_exts)
            ]
            if files:
                latest = max(files, key=os.path.getmtime)
                # Replace the skeleton with the output image in-place
                st.session_state.test_model_output_path = latest
                # Success banner + image
                st.markdown("<div class='success-banner'><span class='success-check'></span>Classification complete</div>", unsafe_allow_html=True)
                output_placeholder.image(latest, caption=None, width=420)
                st.balloons()
            else:
                st.warning("No output images found in tabs/test_model/output.")


if __name__ == "__main__":
    main()
