import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from pathlib import Path
import os
import gc
import pickle
import hashlib

# Import simplified modules
from Bert_config import SimpleBERTEmbeddings
from data_preprocessing import SimpleDataPreprocessor
from Train_models import SimpleEmotionClassifiers
from Model_evaluation import SimpleModelEvaluator

# ===================== CACHING SYSTEM =====================

def initialize_session_state():
    """Initialize all session state variables with caching flags"""
    
    # Step tracking
    if 'current_step' not in st.session_state:
        st.session_state.current_step = 1
    
    # Data caching flags
    cache_keys = [
        'data_loaded', 'data_processed', 'embeddings_generated', 
        'models_trained', 'models_evaluated', 'df', 'X_train', 
        'X_test', 'y_train', 'y_test', 'X_train_embeddings', 
        'X_test_embeddings', 'classifiers', 'bert_embedder', 
        'results', 'preprocessing_options', 'selected_model',
        'data_hash', 'processing_hash', 'model_hash', 'step',
        'best_model_for_prediction', 'show_summary'
    ]
    
    for key in cache_keys:
        if key not in st.session_state:
            if key.endswith('_hash'):
                st.session_state[key] = None
            elif key in ['data_loaded', 'data_processed', 'embeddings_generated', 'models_trained', 'models_evaluated']:
                st.session_state[key] = False
            elif key == 'step':
                st.session_state[key] = 1
            else:
                st.session_state[key] = None

class SmartCache:
    """Smart caching system that handles different data types"""
    
    def __init__(self, cache_dir="streamlit_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
    def _get_cache_path(self, key, data_hash):
        """Generate cache file path"""
        return self.cache_dir / f"{key}_{data_hash}.pkl"
    
    def _calculate_hash(self, data):
        """Calculate hash for any data type"""
        if isinstance(data, pd.DataFrame):
            return hashlib.md5(str(data.values.tobytes()).encode()).hexdigest()[:10]
        elif isinstance(data, np.ndarray):
            return hashlib.md5(data.tobytes()).hexdigest()[:10]
        elif isinstance(data, (list, tuple)):
            return hashlib.md5(str(data).encode()).hexdigest()[:10]
        else:
            return hashlib.md5(str(data).encode()).hexdigest()[:10]
    
    def save_to_cache(self, key, data, dependencies=None):
        """Save data to cache with dependency tracking"""
        try:
            data_hash = self._calculate_hash(data)
            if dependencies:
                dep_hash = self._calculate_hash(dependencies)
                combined_hash = hashlib.md5(f"{data_hash}_{dep_hash}".encode()).hexdigest()[:10]
            else:
                combined_hash = data_hash
            
            cache_path = self._get_cache_path(key, combined_hash)
            
            with open(cache_path, 'wb') as f:
                pickle.dump({
                    'data': data,
                    'timestamp': time.time(),
                    'hash': combined_hash
                }, f)
            
            st.session_state[f"{key}_hash"] = combined_hash
            return True
            
        except Exception as e:
            st.warning(f"Failed to cache {key}: {str(e)}")
            return False
    
    def load_from_cache(self, key, max_age=3600):
        """Load data from cache if valid"""
        try:
            cached_hash = st.session_state.get(f"{key}_hash")
            if not cached_hash:
                return None
            
            cache_path = self._get_cache_path(key, cached_hash)
            
            if not cache_path.exists():
                return None
            
            if time.time() - cache_path.stat().st_mtime > max_age:
                return None
            
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            
            return cached_data['data']
            
        except Exception as e:
            return None

@st.cache_data(ttl=3600, max_entries=3)
def load_and_cache_csv(uploaded_file_bytes, file_name):
    """Cache CSV loading to avoid re-reading"""
    if uploaded_file_bytes is not None:
        file_hash = hashlib.md5(uploaded_file_bytes).hexdigest()
        df = pd.read_csv(pd.io.common.BytesIO(uploaded_file_bytes))
        return df, file_hash
    return None, None

@st.cache_resource(ttl=1800)
def get_cached_bert_embedder(model_name):
    """Cache BERT embedder to avoid reloading"""
    embedder = SimpleBERTEmbeddings(model_name=model_name)
    success = embedder.load_model()
    if success:
        return embedder
    return None

# Initialize everything
initialize_session_state()
cache_manager = SmartCache()

# Essential emotion labels (27 emotions)
EMOTION_LABELS = [
    'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
    'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
    'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
    'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
    'relief', 'remorse', 'sadness', 'surprise', 'neutral'
]

# Team info
TEAM_NAMES = [
    "Zoe Akua Ohene-Ampofo, 22252412",
    "Yvette S. Nerquaye-Tetteh, 22253082", 
    "Theophilus Arthur, 11410587",
    "Suleman Abdul-Razark, 22256374",
    "Steve Afrifa-Yamoah, 22252462",
]

# Logo path fix
def get_logo_path():
    """Get logo path that works both locally and in deployment"""
    possible_paths = [
        "assets/logo.png",
        "logo.png", 
        "Logo_emotion_detection.png",
        "./assets/logo.png"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    return None

LOGO_PATH = get_logo_path()

# Accent gradient colors
ACCENT_START = "#22d3ee"  # teal
ACCENT_MID   = "#a78bfa"  # purple
ACCENT_END   = "#f472b6"  # pink

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="GoEmotions: Emotion Prediction System",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🎭",
)

# ===================== ENHANCED DARK THEME CSS =====================
def inject_css(acc1, acc2, acc3):
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(180deg, #0d1321 0%, #0a0f1c 50%, #070b14 100%) !important;
            }}
            .glass {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.12);
                border-radius: 12px;
                padding: 1rem 1.1rem;
                box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            }}
            .main-header {{
                font-size: 2.2rem;
                font-weight: 800;
                letter-spacing: 0.5px;
                background: linear-gradient(90deg, {acc1}, {acc2});
                -webkit-background-clip: text;
                background-clip: text;
                color: transparent;
                margin: 0.2rem 0 0.2rem 0;
            }}
            .chip {{
                display: inline-block;
                padding: 6px 10px;
                border-radius: 20px;
                background: rgba(255,255,255,0.1);
                border: 1px solid rgba(255,255,255,0.15);
                margin: 4px 6px 0 0;
                font-size: 0.9rem;
                color: #e5e7eb;
            }}
            .stMarkdown h2, .stMarkdown h3 {{ color: #e0e0e0 !important; }}
            [data-testid="stMetric"] {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 10px;
                padding: 0.9rem 0.9rem;
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }}
            [data-testid="stMetric"] [data-testid="stMetricDelta"] {{ color: {acc1} !important; }}
            .stTabs [data-baseweb="tab-list"] {{ gap: 4px; }}
            .stTabs [data-baseweb="tab"] {{
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.15);
                border-radius: 10px;
                padding: 10px 16px;
                color: #d1d5db;
            }}
            .stTabs [aria-selected="true"] {{
                background: rgba(34, 211, 238, 0.2);
                color: #ffffff !important;
                border-color: {acc1};
                box-shadow: 0 2px 8px rgba(0,0,0,0.3);
            }}
            .stButton > button {{
                background: linear-gradient(135deg, {acc1}, {acc2});
                border: none;
                color: white;
                font-weight: 700;
                padding: 0.6rem 1.1rem;
                border-radius: 10px;
                transition: transform 0.1s ease, box-shadow 0.2s ease;
                box-shadow: 0 4px 12px rgba(34, 211, 238, 0.3);
            }}
            .stButton > button:hover {{
                transform: translateY(-1px);
                box-shadow: 0 6px 16px rgba(34, 211, 238, 0.4);
            }}
            .stTextInput > div > div input, textarea {{
                background: rgba(255,255,255,0.1) !important;
                border: 1px solid rgba(255,255,255,0.2) !important;
                color: #e5e7eb !important;
                border-radius: 8px !important;
            }}
            section[data-testid="stSidebar"] {{
                background: rgba(17,24,39,0.95);
                border-right: 1px solid rgba(255,255,255,0.1);
            }}
            .stDataFrame div[role="table"] {{
                background: rgba(255,255,255,0.05) !important;
                border-radius: 10px !important;
                border: 1px solid rgba(255,255,255,0.15) !important;
            }}
            .footer {{ text-align: center; color: #9ca3af; padding: 18px; }}
            .logo-ring {{
                position: relative;
                width: 58px; height: 58px; border-radius: 50%;
                box-shadow: 0 0 0 2px {acc1};
                overflow: hidden;
                background: rgba(255,255,255,0.1);
            }}
            .logo-ring img {{ width: 100%; height: 100%; object-fit: contain; }}
            .cache-indicator {{
                background: rgba(34, 211, 238, 0.2);
                border: 1px solid #22d3ee;
                border-radius: 8px;
                padding: 8px 12px;
                margin: 4px 0;
                color: #22d3ee;
                font-size: 0.9rem;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

# Initialize accent colors
for key, default in [
    ("acc_start", ACCENT_START),
    ("acc_mid", ACCENT_MID),
    ("acc_end", ACCENT_END),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Apply CSS
inject_css(st.session_state.acc_start, st.session_state.acc_mid, st.session_state.acc_end)

# Memory cleanup function
def clear_memory():
    """Clear memory and GPU cache"""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

# Initialize components with caching
@st.cache_resource
def get_components():
    preprocessor = SimpleDataPreprocessor()
    classifiers = SimpleEmotionClassifiers()
    evaluator = SimpleModelEvaluator()
    return preprocessor, classifiers, evaluator

preprocessor, classifiers, evaluator = get_components()

# ===================== HEADER =====================
header_container = st.container()
with header_container:
    cols = st.columns([0.12, 0.88])
    with cols[0]:
        if LOGO_PATH:
            st.markdown("<div class='logo-ring'>", unsafe_allow_html=True)
            st.image(LOGO_PATH, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)
        else:
            st.write("")
    with cols[1]:
        st.markdown('<div class="main-header">GoEmotions: Multi-Label Emotion Prediction</div>', unsafe_allow_html=True)
        chips = " ".join([f"<span class='chip'>{name}</span>" for name in TEAM_NAMES])
        st.markdown(f"<div>{chips}</div>", unsafe_allow_html=True)
        st.markdown(
            "<div style='color:#9ca3af; margin-top:6px;'>"
            "<strong>Project:</strong> BERT embeddings + Balanced Sampling + Binary Classification &nbsp;|&nbsp; "
            "<strong>Dataset:</strong> GoEmotions (27 labels, perfectly balanced)"
            "</div>",
            unsafe_allow_html=True,
        )

st.markdown("<div class='glass' style='margin-top:0.6rem'></div>", unsafe_allow_html=True)

# ===================== CACHE STATUS INDICATOR =====================
def show_cache_status():
    """Show cache status in a nice indicator"""
    cached_items = []
    if st.session_state.get('data_loaded', False):
        cached_items.append("Data")
    if st.session_state.get('data_processed', False):
        cached_items.append("Balanced Sampling")
    if st.session_state.get('embeddings_generated', False):
        cached_items.append("Embeddings")
    if st.session_state.get('models_trained', False):
        cached_items.append("Models")
    if st.session_state.get('models_evaluated', False):
        cached_items.append("Evaluation")
    
    if cached_items:
        st.markdown(
            f"<div class='cache-indicator'>🗄️ Cached: {', '.join(cached_items)}</div>",
            unsafe_allow_html=True
        )

show_cache_status()

# ===================== TABS =====================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Data Overview",
    "Balanced Sampling", 
    "BERT Embeddings",
    "Model Training",
    "Model Evaluation",
    "Live Prediction",
    "Project Summary"
])

# Progress indicator
progress_steps = ["Upload Data", "Balance Data", "Train Model", "Evaluate Model", "Predict Emotion"]
current_step = st.session_state.get('step', 1)

# ===================== TAB 1: DATA OVERVIEW =====================
with tab1:
    st.header("Step 1: Upload Your Data")
    
    # Check for cached data first
    if st.session_state.get('data_loaded', False) and 'df' in st.session_state and st.session_state.df is not None:
        st.success("📋 Using cached data!")
        df = st.session_state.df
        
        # Show basic info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Samples", f"{len(df):,}")
        with col2:
            st.metric("Emotions", len(EMOTION_LABELS))
        with col3:
            avg_length = df['text'].str.len().mean()
            st.metric("Avg Text Length", f"{avg_length:.0f}")
        
        # Show emotion distribution
        st.subheader("Current Emotion Distribution (IMBALANCED)")
        emotion_counts = {}
        for emotion in EMOTION_LABELS:
            if emotion in df.columns:
                emotion_counts[emotion] = (df[emotion] == 1).sum()
        
        # Show top 10 and bottom 10
        sorted_emotions = sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)

        # Get the emotion counts as a pandas Series
        emotion_counts = pd.Series(emotion_counts)

        # Step 1: Convert the dictionary to a DataFrame
        # This explicitly creates a DataFrame with 'emotion' and 'count' columns
        df_emotion_counts = pd.DataFrame(
            list(emotion_counts.items()),
            columns=['emotion', 'count']
        )

        # Sort the counts in descending order
        sorted_emotion_counts = emotion_counts.sort_values(ascending=False)

        # Create the plot with Matplotlib
        fig, ax = plt.subplots(figsize=(10, 6))

        # Use the sorted data to plot the bar chart
        ax.bar(sorted_emotion_counts.index, sorted_emotion_counts.values, color='lightgreen')

        # Add plot titles and labels
        ax.set_title('Emotion Distribution (Sorted)', fontsize=16, color='darkblue')
        ax.set_xlabel('Emotions', fontsize=12, color='white')
        ax.set_ylabel('Count', fontsize=12, color='white')

        # Rotate x-axis labels for better readability
        plt.xticks(rotation=90, ha='right', color='black')
        plt.yticks(color='black')

        # Adjust plot layout
        plt.tight_layout()

        # Display the Matplotlib figure in Streamlit
        st.pyplot(fig)

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Most Common Emotions:**")
            for emotion, count in sorted_emotions[:5]:
                pct = (count / len(df)) * 100
                st.write(f"• {emotion.title()}: {count:,} ({pct:.1f}%)")

        with col2:
            st.write("**Rarest Emotions:**")
            for emotion, count in sorted_emotions[-5:]:
                pct = (count / len(df)) * 100
                st.write(f"• {emotion.title()}: {count:,} ({pct:.1f}%)")

        st.info("🎯 **Next Step**: Use balanced sampling to fix this imbalance!")

        # Show preview
        st.subheader("Data Preview")
        st.dataframe(df.head(), use_container_width=True)
        
    else:
        uploaded_file = st.file_uploader(
            "Upload GoEmotions CSV file",
            type=['csv'],
            help="Upload your CSV file with 'text' column and emotion labels"
        )
        
        if uploaded_file:
            # Use cached loading
            file_bytes = uploaded_file.getvalue()
            df, file_hash = load_and_cache_csv(file_bytes, uploaded_file.name)
            
            if df is not None:
                st.session_state.df = df
                st.session_state.data_loaded = True
                st.session_state.data_hash = file_hash
                st.session_state.step = max(st.session_state.step, 2)
                
                st.success("✅ Data loaded and cached!")
                st.rerun()

# ===================== TAB 2: BALANCED SAMPLING =====================
with tab2:
    st.header("Step 2: Balanced Emotion Sampling (NEW APPROACH)")
    
    if not st.session_state.get('data_loaded', False):
        st.warning("Please upload data first")
    else:
        df = st.session_state.df
        
        # Check for cached processed data
        if st.session_state.get('data_processed', False) and all(key in st.session_state for key in ['X_train', 'X_test', 'y_train', 'y_test']):
            st.success("🚀 Using cached balanced data!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Training Samples", f"{len(st.session_state.X_train):,}")
            with col2:
                st.metric("Test Samples", f"{len(st.session_state.X_test):,}")
            with col3:
                st.metric("Status", "Perfectly Balanced")
                
        else:
            # Explain the new approach
            st.info("🎯 **NEW STRATEGY**: Instead of complex oversampling, we'll use equal samples per emotion!")
            
            col1, col2 = st.columns([2, 1])
            with col1:
                st.write("**Benefits of Balanced Sampling:**")
                st.write("✅ **Perfect balance** - Every emotion gets equal training attention")
                st.write("✅ **Simpler approach** - No complex oversampling algorithms")  
                st.write("✅ **Better performance** - Consistent accuracy across all emotions")
                st.write("✅ **Faster training** - Manageable dataset size")
                st.write("✅ **Memory efficient** - Works within your hardware constraints")
            
            with col2:
                # Calculate emotion counts
                emotion_counts = {}
                for emotion in EMOTION_LABELS:
                    if emotion in df.columns:
                        emotion_counts[emotion] = (df[emotion] == 1).sum()
                
                min_available = min(emotion_counts.values())
                max_available = max(emotion_counts.values())
                
                st.metric("Min Available", f"{min_available:,}")
                st.metric("Max Available", f"{max_available:,}")
                st.metric("Imbalance Ratio", f"{min_available/max_available:.2f}")
            
            # Show preprocessing options
            preprocessing_options = preprocessor.show_interactive_preprocessing_options()
            
            col1, col2 = st.columns(2)
            with col1:
                use_balanced_sampling = st.checkbox("Enable Balanced Sampling", value=True, 
                    help="Sample equal amounts from each emotion for perfect balance")
            with col2:
                total_target = st.slider("Total Dataset Size", 10000, 30000, 20000, step=1000,
                    help="Total samples across all emotions (will be divided equally)")
            
            samples_per_emotion = total_target // len(EMOTION_LABELS)
            st.info(f"Target: {samples_per_emotion:,} samples per emotion = {total_target:,} total samples")
            
            if st.button("Create Balanced Dataset", type="primary"):
                with st.spinner("Creating perfectly balanced dataset..."):
                    # Process data with balanced sampling
                    X_train, X_test, y_train, y_test = preprocessor.process_data(
                        df, 
                        sample_size=total_target,
                        preprocessing_options=preprocessing_options,
                        use_balanced_sampling=use_balanced_sampling
                    )
                    
                    if X_train is not None:
                        # Cache the results
                        st.session_state.X_train = X_train
                        st.session_state.X_test = X_test
                        st.session_state.y_train = y_train
                        st.session_state.y_test = y_test
                        st.session_state.data_processed = True
                        st.session_state.preprocessing_options = preprocessing_options
                        st.session_state.step = max(st.session_state.step, 3)
                        
                        st.success("✅ Balanced dataset created and cached!")
                        st.rerun()

# ===================== TAB 3: BERT EMBEDDINGS =====================
with tab3:
    st.header("Step 3: BERT Embeddings")
    
    if not st.session_state.get('data_processed', False):
        st.warning("Please create balanced dataset first")
    else:
        # Model selection with caching
        selected_model = st.selectbox(
            "Choose BERT Model:",
            ['bert-base-uncased', 'bert-large-uncased'],
            index=0
        )
        
        # Check for cached embeddings
        if (st.session_state.get('embeddings_generated', False) and 
            'X_train_embeddings' in st.session_state and 
            'X_test_embeddings' in st.session_state and
            st.session_state.get('selected_model') == selected_model):
            
            st.success("🚀 Using cached embeddings!")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Model", selected_model)
            with col2:
                st.metric("Embedding Dim", st.session_state.X_train_embeddings.shape[1])
            with col3:
                st.metric("Status", "Ready for Training")
                
        else:
            if st.button("Generate Embeddings", type="primary"):
                with st.spinner("Generating embeddings with caching..."):
                    # Get cached BERT embedder
                    bert_embedder = get_cached_bert_embedder(selected_model)
                    
                    if bert_embedder:
                        # Generate embeddings
                        X_train_embeddings = bert_embedder.generate_embeddings(st.session_state.X_train)
                        X_test_embeddings = bert_embedder.generate_embeddings(st.session_state.X_test)
                        
                        if X_train_embeddings is not None and X_test_embeddings is not None:
                            # Cache results
                            st.session_state.X_train_embeddings = X_train_embeddings
                            st.session_state.X_test_embeddings = X_test_embeddings
                            st.session_state.bert_embedder = bert_embedder
                            st.session_state.selected_model = selected_model
                            st.session_state.embeddings_generated = True
                            st.session_state.step = max(st.session_state.step, 4)
                            
                            st.success("✅ Embeddings generated and cached!")
                            st.rerun()

# ===================== TAB 4: MODEL TRAINING =====================
with tab4:
    st.header("Step 4: Model Training")
    
    if not st.session_state.get('embeddings_generated', False):
        st.warning("Please generate embeddings first")
    else:
        # Check for cached models
        if st.session_state.get('models_trained', False) and 'classifiers' in st.session_state:
            st.success("🤖 Using cached trained models!")
            
            st.subheader("Training Summary")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Models Trained", "2")
            with col2:
                st.metric("Training Samples", f"{len(st.session_state.X_train_embeddings):,}")
            with col3:
                st.metric("Status", "Ready for Evaluation")
                
        else:
            st.info("🎯 **Training on Perfectly Balanced Data**: Each emotion gets equal attention!")
            
            if st.button("Train Models", type="primary"):
                with st.spinner("Training models on balanced data..."):
                    # Train models
                    nb_success = classifiers.train_naive_bayes(
                        st.session_state.X_train_embeddings, 
                        st.session_state.y_train
                    )
                    rf_success = classifiers.train_random_forest(
                        st.session_state.X_train_embeddings, 
                        st.session_state.y_train
                    )
                    
                    if nb_success or rf_success:
                        # Cache results
                        st.session_state.classifiers = classifiers
                        st.session_state.models_trained = True
                        st.session_state.step = max(st.session_state.step, 5)
                        
                        st.success("✅ Models trained on balanced data and cached!")
                        st.rerun()

# ===================== TAB 5: MODEL EVALUATION =====================
with tab5:
    st.header("Step 5: Model Evaluation")
    
    if not st.session_state.get('models_trained', False):
        st.warning("Please train models first")
    else:
        # Check for cached evaluation
        if st.session_state.get('models_evaluated', False) and 'results' in st.session_state:
            st.success("📊 Using cached evaluation results!")
            
            # Display cached results
            results = st.session_state.results
            summary_data = evaluator.display_performance_summary(results)
            
            if summary_data and len(summary_data) > 1:
                best_model_data = max(summary_data, key=lambda x: float(x['Accuracy'].replace('±','').rstrip('%')))
                best_model_name = best_model_data['Model'].lower().replace(' ', '_')
                st.session_state.best_model_for_prediction = best_model_name
                
        else:
            st.info("📈 **Evaluating on Balanced Test Data**: Expect consistent performance across ALL emotions!")
            
            if st.button("Evaluate Models", type="primary"):
                with st.spinner("Evaluating models on balanced test data..."):
                    # Evaluate models
                    results = evaluator.evaluate_models(
                        st.session_state.classifiers,
                        st.session_state.X_test_embeddings,
                        st.session_state.y_test
                    )
                    
                    if results:
                        # Cache results
                        st.session_state.results = results
                        st.session_state.models_evaluated = True
                        
                        # Determine best model
                        summary_data = evaluator.display_performance_summary(results)
                        if summary_data and len(summary_data) > 1:
                            best_model_data = max(summary_data, key=lambda x: float(x['Accuracy'].replace('±','').rstrip('%')))
                            best_model_name = best_model_data['Model'].lower().replace(' ', '_')
                            st.session_state.best_model_for_prediction = best_model_name
                        
                        st.success("✅ Evaluation completed on balanced data and cached!")
                        st.rerun()

# ===================== TAB 6: LIVE PREDICTION =====================
with tab6:
    st.header("Step 6: Live Prediction")
    
    if not st.session_state.get('models_evaluated', False):
        st.warning("Please complete evaluation first")
    else:
        best_model = st.session_state.get('best_model_for_prediction', 'random_forest')
        
        # Single text prediction
        st.subheader("Single Text Prediction")

        # Add custom CSS to style the text area (input text + placeholder)
        st.markdown(
            """
            <style>
            /* Ensure text area has a white background and black text */
            .stTextArea textarea {
                color: black !important;  /* Set text color to black */
                background-color: white !important;  /* Force background color to white */
                border: 1px solid #ccc;  /* Optional: set border */
                padding: 10px;  /* Optional: padding to improve appearance */
                font-size: 16px;  /* Optional: set font size */
                width: 100%;  /* Ensure text area fills the width */
            }
            /* Ensure placeholder text is black */
            .stTextArea textarea::placeholder {
                color: black !important;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        text_input = st.text_area(
            "Enter text to analyze:",
            placeholder="Type something like: 'I'm so excited about this!' or 'This makes me angry.'"
        )
        
        if st.button("Predict Emotions", type="primary"):
            if text_input.strip():
                with st.spinner("Analyzing emotions..."):
                    results = st.session_state.classifiers.predict_single_text_with_attention(
                        text_input, 
                        st.session_state.bert_embedder,
                        model_type=best_model,
                        threshold=0.3,
                        return_attention=True
                    )
                    
                    if results and 'top_3_emotions' in results:
                        st.subheader("Top 3 Predicted Emotions")
                        
                        for i, (emotion, prob) in enumerate(zip(results['top_3_emotions'], results['top_3_probabilities'])):
                            rank = i + 1
                            confidence_pct = prob * 100
                            
                            st.write(f"**#{rank} {emotion.title()}**: {confidence_pct:.1f}% confidence")
                            st.progress(prob, text=f"{emotion.title()}: {confidence_pct:.1f}%")

# ===================== TAB 7: PROJECT SUMMARY =====================
with tab7:
    st.header("Project Summary")
    
    st.write("## GoEmotions: Multi-Label Emotion Prediction System")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🔧 Technical Stack")
        st.write("- **BERT**: bert-base-uncased for embeddings")
        st.write("- **Models**: Naive Bayes + Random Forest")
        st.write("- **Dataset**: GoEmotions (27 emotions)")
        st.write("- **NEW: Balanced Sampling**: Equal samples per emotion")
        st.write("- **Caching**: Smart session management")
        
        st.write("### 🎯 Key Innovation")
        st.success("**Balanced Sampling Strategy**: Instead of complex oversampling, we use equal samples per emotion for perfect balance!")
        
    with col2:
        st.write("### 📊 Expected Performance")
        st.write("**With Balanced Sampling:**")
        st.write("- **All emotions**: 60-80% F1 (consistent)")
        st.write("- **No more rare emotion problem**")
        st.write("- **Faster training**: Manageable dataset size")
        st.write("- **Better predictions**: Equal attention to all emotions")
        
        if 'results' in st.session_state and st.session_state.results is not None:
            st.write("### 📈 Actual Results")
            for model_name, metrics in st.session_state.results.items():
                accuracy = metrics.get('hamming_accuracy', 0) * 100
                f1 = metrics.get('macro_f1', 0) * 100
                st.write(f"**{model_name.replace('_', ' ').title()}**:")
                st.write(f"- Accuracy: {accuracy:.1f}%")
                st.write(f"- F1-Score: {f1:.1f}%")

# ===================== ENHANCED SIDEBAR WITH CACHE STATUS =====================
with st.sidebar:
    st.header("🗄️ System Status")
    
    status_items = [
        ("Data Loaded", st.session_state.get('data_loaded', False)),
        ("Balanced Sampling", st.session_state.get('data_processed', False)),
        ("Embeddings Generated", st.session_state.get('embeddings_generated', False)),
        ("Models Trained", st.session_state.get('models_trained', False)),
        ("Models Evaluated", st.session_state.get('models_evaluated', False))
    ]
    
    for label, status in status_items:
        if status:
            st.success(f"✅ {label}")
        else:
            st.error(f"❌ {label}")
    
    # Progress indicator
    completed_steps = sum(1 for _, status in status_items if status)
    total_steps = len(status_items)
    progress = completed_steps / total_steps
    
    st.subheader("Overall Progress")
    st.progress(progress)
    st.write(f"**{completed_steps}/{total_steps} steps completed**")
    
    st.divider()
    
    # Show balanced sampling benefits
    st.subheader("🎯 Balanced Sampling Benefits")
    st.success("✅ Perfect emotion balance")
    st.success("✅ Consistent performance")
    st.success("✅ Memory efficient")
    st.success("✅ Faster training")
    st.success("✅ No rare emotion problem")
    
    st.divider()
    
    # Cache management
    st.subheader("Cache Management")
    
    if st.button("Clear All Cache"):
        # Clear session state cache flags
        for key in ['data_loaded', 'data_processed', 'embeddings_generated', 'models_trained', 'models_evaluated']:
            st.session_state[key] = False
        
        # Clear cache directory
        try:
            cache_manager.clear_old_cache(max_age=0)
            st.success("Cache cleared!")
        except:
            st.warning("Could not clear file cache")
        
        clear_memory()
        st.rerun()
    
    if st.button("Start Over"):
        # Clear everything and restart
        for key in list(st.session_state.keys()):
            if key not in ['acc_start', 'acc_mid', 'acc_end']:
                del st.session_state[key]
        
        initialize_session_state()
        clear_memory()
        st.rerun()

# Footer
st.divider()
st.markdown(
    "<div class='footer'>"
    "<p><strong>GoEmotions: Advanced Emotion Detection System</strong></p>"
    "<p>BERT-based Multi-Label Emotion Classification with Balanced Sampling</p>"
    "</div>",
    unsafe_allow_html=True
)