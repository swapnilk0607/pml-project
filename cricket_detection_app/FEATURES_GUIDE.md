# 🎯 Feature Guide - Cricket Detection UI

## Main Features Overview

### 🔴 Feature 1: Multi-Image Upload Button
**Location**: Upload & Detect Tab - Top Section

```
📁 Current images in directory: 5

┌────────────────────────────────────┐
│ Choose cricket images to upload    │  [🗑️ Clear All]
│ [Browse] [Browse] [Browse] ...     │
└────────────────────────────────────┘

✅ 3 image(s) selected for upload.

[🚀 Save & Run Model]
```

**What it does:**
- Browse and select multiple images in one go
- Shows how many images are currently in the directory
- Clear button to reset all data
- One-click model execution

**Technical Details:**
- Uses Streamlit's `file_uploader()` with `accept_multiple_files=True`
- Saves to persistent `input_images/` directory
- Supports PNG, JPG, JPEG formats

---

### 🔴 Feature 2: Image Results Display
**Location**: Upload & Detect Tab - After Running Model

```
2. Detection Results
Displaying results for 3 processed image(s)

┌─────────────────────────────────────┐
│  [Image Gallery] [Detailed Results] │
└─────────────────────────────────────┘

IMAGE GALLERY VIEW:
┌──────────┬──────────┬──────────┐
│ Image 1  │ Image 2  │ Image 3  │
│ (boxes)  │ (boxes)  │ (boxes)  │
│          │          │          │
│ 🕐 14:30 │ 🕐 14:31 │ 🕐 14:32 │
└──────────┴──────────┴──────────┘

DETAILED RESULTS VIEW:
📄 image1.jpg - 5 objects detected
   ├─ Batsman: 0.95
   ├─ Ball: 0.87
   └─ ...
```

**What it does:**
- Displays uploaded images with annotated bounding boxes
- 3-column grid for easy viewing
- Toggle between gallery and detailed JSON views
- Shows processing timestamp

---

### 🔴 Feature 3: Graphical Analytics Dashboard
**Location**: Graphical Analytics Tab (Sidebar)

```
📊 Model Performance Analytics

┌────────────────────────────────────────┐
│ 📷 10  │ 🎯 156  │ ⭐ 92%  │ 🔝 99%  │
│ Images │ Objects │ Avg     │ Max     │
│Processed│Detected│Confidence│Confidence
└────────────────────────────────────────┘
```

#### Sub-Feature 3a: Distribution Tab
```
[📊 Distribution] [📈 Confidence] [🎯 Class] [📋 Data]

Object Class Distribution

Bar Chart:              Pie Chart:
Batsman    ████████    Batsman  45%
Ball       ████        Ball     20%
Bowler     ███         Bowler   18%
Stump      ███         Stump    10%
```

#### Sub-Feature 3b: Confidence Analysis Tab
```
Confidence Score Analysis

HISTOGRAM:
  Freq │
   15  │     ╱╲
   10  │    ╱  ╲
    5  │   ╱    ╲
    0  │──────────── Confidence Score
       0.7  0.8  0.9  1.0

STATISTICS:
- Mean: 0.892
- Median: 0.899
- Std Dev: 0.045
- Min: 0.701
- Max: 0.995
```

#### Sub-Feature 3c: Class Breakdown Tab
```
Detailed Class Analysis

┌──────────┬────────┬────┬────┬────┐
│ Class    │ Count  │Avg │Min │Max │
├──────────┼────────┼────┼────┼────┤
│ Batsman  │ 45     │0.91│0.78│0.97│
│ Ball     │ 30     │0.85│0.71│0.95│
│ Bowler   │ 27     │0.88│0.72│0.99│
└──────────┴────────┴────┴────┴────┘

Box Plot:              Scatter Plot:
Confidence             1.0 ┤ • •
by Class               0.8 ┤• ••
                       0.6 ┤
```

#### Sub-Feature 3d: Data Table Tab
```
Raw Detection Data

┌──────┬────────┬────────────┐
│Class │Confidence│Bbox      │
├──────┼────────┼────────────┤
│ Bats │ 0.95   │ [50,80...]│
│ Ball │ 0.87   │ [120,45...]
│ Bowl │ 0.92   │ [90,60...]│
└──────┴────────┴────────────┘

[📥 Download Results as CSV]
```

---

## 📱 UI Layout Structure

```
┌─────────────────────────────────────────────┐
│           🏏 CRICKET OBJECT DETECTION        │
│    Upload match images to detect players    │
├─────────────────┬──────────────────────────┤
│ SIDEBAR         │      MAIN CONTENT        │
│                 │                          │
│ Navigate:       │  [Upload & Detect]      │
│ • Upload &      │                          │
│   Detect        │  or                      │
│ • Graphical     │                          │
│   Analytics     │  [Graphical Analytics]  │
│                 │                          │
└─────────────────┴──────────────────────────┘
```

---

## 🎨 Color Scheme & Emojis

| Element | Emoji | Color | Purpose |
|---------|-------|-------|---------|
| Upload | 📤 | Blue | Action indicator |
| Images | 📁 | Gray | Storage indicator |
| Delete | 🗑️ | Red | Destructive action |
| Run | 🚀 | Green | Start action |
| Success | ✅ | Green | Confirmation |
| Error | ❌ | Red | Alert |
| Analytics | 📊 | Purple | Data/Charts |
| Data | 📋 | Orange | Tables/Info |
| Download | 📥 | Blue | Export |
| Stats | ⭐ | Yellow | Metrics |

---

## 🔄 User Workflow

```
START
  │
  ├─> Sidebar: Click "Upload & Detect"
  │
  ├─> Upload Section
  │   ├─> View current images count
  │   ├─> Select multiple images
  │   └─> Click "Save & Run Model"
  │
  ├─> Processing
  │   └─> 📤 Uploading images and running detection pipeline...
  │
  ├─> Results Display
  │   ├─> Image Gallery (3-column grid with boxes)
  │   └─> Detailed Results (JSON expansion)
  │
  ├─> Analytics
  │   ├─> Sidebar: Click "Graphical Analytics"
  │   ├─> View 4 Tabs:
  │   │   ├─> Distribution (Bar + Pie)
  │   │   ├─> Confidence Analysis (Histogram + Stats)
  │   │   ├─> Class Breakdown (Tables + Plots)
  │   │   └─> Data Table (Raw + CSV Export)
  │   │
  │   └─> Optional: Download CSV
  │
  └─> END
```

---

## 💾 Data Flow

```
User Uploads Files
        ↓
[Save to input_images/]
        ↓
[Run Detection Model]
        ↓
[Results Object]
{
    filename: "image.jpg",
    detections: [
        {class: "Batsman", confidence: 0.95, bbox: [...]},
        {class: "Ball", confidence: 0.87, bbox: [...]}
    ],
    processed_at: "14:30:45"
}
        ↓
[Store in Session State]
        ↓
[Display on UI]
├─> Image with boxes
├─> JSON details
└─> Analytics charts
```

---

## 🎯 Key Buttons & Actions

| Button | Location | Action | Result |
|--------|----------|--------|--------|
| 📤 Browse | Upload section | Select images | Images added to upload queue |
| 🗑️ Clear All | Upload section | Clear directory | All images removed, results reset |
| 🚀 Save & Run | Upload section | Process images | Images saved and model runs |
| 📊 Distribution | Analytics tab | View | Class distribution charts |
| 📈 Confidence | Analytics tab | View | Confidence score analysis |
| 🎯 Class | Analytics tab | View | Per-class breakdown |
| 📋 Data | Analytics tab | View | Raw data table |
| 📥 Download | Data tab | Export | CSV file downloaded |

---

## 🔧 Settings & Customization

All configurations are in the code:

```python
# Page Config
st.set_page_config(
    page_title="Cricket Object Detection",  # Change this
    page_icon="🏏",                         # Change this
    layout="wide"
)

# Supported formats
type=['png', 'jpg', 'jpeg']                 # Add more if needed

# Constants
INPUT_DIR = "input_images"                  # Change path if needed

# Histogram bins
bins=15                                     # Adjust chart granularity
```

---

## 📊 Analytics Explained

### Distribution Tab
Shows what objects were detected:
- **Bar Chart**: Count of each class
- **Pie Chart**: Percentage distribution

### Confidence Tab
Shows how confident the model was:
- **Histogram**: Spread of confidence scores
- **Mean**: Average confidence
- **Median**: Middle confidence value
- **Std Dev**: Variance in confidence

### Class Breakdown Tab
Detailed analysis per object class:
- **Table**: Statistics for each class
- **Box Plot**: Range and quartiles
- **Scatter**: Individual detection scores

### Data Table
Raw detection information:
- All detected objects with confidence
- Export capability for further analysis

---

## ✨ Best Practices for Using

1. **Organize Images**: Keep related images in batches
2. **Monitor Confidence**: Watch for low scores (< 0.7)
3. **Check Distribution**: Ensure balanced detection
4. **Export Data**: Save results for records
5. **Clear When Done**: Reset directory between projects

---

**Quick Tip**: The app remembers results even if you navigate between tabs. Use this to compare detection results with analytics side-by-side!

