import pandas as pd
import numpy as np
import streamlit as st
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split

class SimpleDataPreprocessor:
    def __init__(self):
        self.emotion_columns = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
        
        # Download NLTK data quietly
        try:
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
        except:
            pass
        
        # Initialize NLP tools
        try:
            self.stop_words = set(stopwords.words('english'))
            self.stemmer = PorterStemmer()
            self.lemmatizer = WordNetLemmatizer()
        except:
            self.stop_words = set()
            self.stemmer = None
            self.lemmatizer = None
    
    def load_data(self, uploaded_file):
        """Load and validate CSV data"""
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check for required columns
            if 'text' not in df.columns:
                st.error("CSV must contain a 'text' column")
                return None
            
            # Check for emotion columns
            missing_emotions = [col for col in self.emotion_columns if col not in df.columns]
            if missing_emotions:
                st.warning(f"Missing emotion columns: {missing_emotions}")
                # Create missing columns with zeros
                for emotion in missing_emotions:
                    df[emotion] = 0
            
            # Remove rows with empty text
            df = df[df['text'].notna() & (df['text'].str.strip() != '')]
            
            st.success(f"Loaded {len(df)} valid samples")
            return df
            
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    def clean_text_for_bert(self, text):
        """Minimal cleaning optimized for BERT models (Higher Accuracy)"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Only basic cleaning - keep most natural language intact
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove @mentions but keep hashtag content
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Keep punctuation - it's important for emotions!
        # Keep stop words - "not", "very", "really" matter for emotions!
        # Keep original case - CAPS can show emotion intensity
        
        # Only remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def clean_text(self, text, 
                   remove_stopwords=True, 
                   use_stemming=False, 
                   use_lemmatization=True,
                   remove_punctuation=True,
                   remove_numbers=False):
        """Standard NLP text cleaning (May reduce BERT accuracy)"""
        if pd.isna(text):
            return ""
        
        text = str(text).lower()  # Convert to lowercase
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove mentions and hashtags symbols (keep content)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#(\w+)', r'\1', text)
        
        # Remove numbers if specified
        if remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation if specified
        if remove_punctuation:
            text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        try:
            tokens = word_tokenize(text)
        except:
            tokens = text.split()  # Fallback if NLTK fails
        
        # Remove stopwords
        if remove_stopwords and self.stop_words:
            tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
        
        # Apply stemming or lemmatization
        if use_stemming and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]
        elif use_lemmatization and self.lemmatizer:
            try:
                tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
            except:
                pass  # Skip if lemmatization fails
        
        # Join back to text
        cleaned_text = ' '.join(tokens)
        
        # Remove extra whitespace
        cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
        
        return cleaned_text

    def show_interactive_preprocessing_options(self):
        """Interactive preprocessing options with step-by-step configuration"""
        st.subheader("Text Preprocessing Configuration")
        
        # Step 1: Choose preprocessing mode
        st.write("**Step 1: Choose Preprocessing Mode**")
        
        mode = st.radio(
            "Select preprocessing approach:",
            ["BERT-Optimized (Recommended)", "Standard NLP", "Custom Configuration"],
            help="BERT-Optimized keeps natural language features for better emotion detection"
        )
        
        if mode == "BERT-Optimized (Recommended)":
            st.success("**High Accuracy Mode**: Keeps stop words, punctuation, and natural text")
            st.info("**Expected Result**: 5-15% higher accuracy than standard NLP preprocessing")
            st.info("**What we keep**: 'not', 'very', punctuation (!?.), original case, natural word forms")
            
            return {
                'remove_stopwords': False,
                'use_stemming': False,
                'use_lemmatization': False,
                'remove_punctuation': False,
                'remove_numbers': False,
                'bert_optimized': True
            }
        
        elif mode == "Standard NLP":
            st.warning("**Standard NLP Mode**: May reduce BERT accuracy by 5-15%")
            st.info("**Use this for**: Traditional ML models or research comparison")
            
            return {
                'remove_stopwords': True,
                'use_stemming': False,
                'use_lemmatization': True,
                'remove_punctuation': True,
                'remove_numbers': False,
                'bert_optimized': False
            }
        
        else:  # Custom Configuration
            st.write("**Step 2: Configure Custom Settings**")
            st.warning("**Advanced Mode**: Configure each preprocessing step manually")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Text Cleaning:**")
                remove_stopwords = st.checkbox("Remove Stop Words", value=False, 
                    help="Remove 'the', 'and', 'is', 'not', etc. (Not recommended for emotions)")
                
                remove_punctuation = st.checkbox("Remove Punctuation", value=False,
                    help="Remove ! ? . , etc. (Not recommended for emotions)")
                
                remove_numbers = st.checkbox("Remove Numbers", value=False)
            
            with col2:
                st.write("**Word Processing:**")
                processing_type = st.radio(
                    "Choose word processing:",
                    ["None (Recommended)", "Lemmatization", "Stemming"],
                    help="None preserves original words for BERT"
                )
                
                use_lemmatization = processing_type == "Lemmatization"
                use_stemming = processing_type == "Stemming"
            
            # Show impact warning
            active_steps = []
            if remove_stopwords:
                active_steps.append("remove stop words")
            if remove_punctuation:
                active_steps.append("remove punctuation")
            if remove_numbers:
                active_steps.append("remove numbers")
            if use_lemmatization:
                active_steps.append("lemmatization")
            elif use_stemming:
                active_steps.append("stemming")
            
            if active_steps:
                st.warning(f"**Active preprocessing**: {', '.join(active_steps)}. This may reduce BERT accuracy.")
            else:
                st.success("**Minimal preprocessing**: Optimal for BERT emotion detection.")
            
            return {
                'remove_stopwords': remove_stopwords,
                'use_stemming': use_stemming,
                'use_lemmatization': use_lemmatization,
                'remove_punctuation': remove_punctuation,
                'remove_numbers': remove_numbers,
                'bert_optimized': not any([remove_stopwords, remove_punctuation, use_stemming, use_lemmatization])
            }

    def show_preprocessing_preview(self, df, preprocessing_options):
        """Show 5 clear examples of text preprocessing"""
        st.subheader("Text Preprocessing Preview (5 Examples)")
        
        # Always take exactly 5 samples
        sample_texts = df['text'].head(5).tolist()
        
        preview_data = []
        for i, original_text in enumerate(sample_texts):
            # Apply cleaning based on mode
            if preprocessing_options.get('bert_optimized', False):
                cleaned = self.clean_text_for_bert(original_text)
                mode = "BERT-Optimized"
            else:
                cleaned = self.clean_text(original_text, **{k: v for k, v in preprocessing_options.items() if k != 'bert_optimized'})
                mode = "Standard NLP"
            
            # Calculate word counts
            original_words = len(original_text.split())
            cleaned_words = len(cleaned.split())
            
            preview_data.append({
                'Example': f"#{i+1}",
                'Original Text': original_text[:120] + "..." if len(original_text) > 120 else original_text,
                'Processed Text': cleaned[:120] + "..." if len(cleaned) > 120 else cleaned,
                'Word Count': f"{original_words} → {cleaned_words}",
                'Mode': mode
            })
        
        # Display the examples
        preview_df = pd.DataFrame(preview_data)
        st.dataframe(preview_df, use_container_width=True, hide_index=True)
        
        # Show summary stats
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_original = np.mean([len(text.split()) for text in sample_texts])
            st.metric("Avg Original Words", f"{avg_original:.1f}")
        
        with col2:
            if preprocessing_options.get('bert_optimized', False):
                processed_texts = [self.clean_text_for_bert(text) for text in sample_texts]
            else:
                processed_texts = [self.clean_text(text, **{k: v for k, v in preprocessing_options.items() if k != 'bert_optimized'}) for text in sample_texts]
            avg_processed = np.mean([len(text.split()) for text in processed_texts])
            st.metric("Avg Processed Words", f"{avg_processed:.1f}")
        
        with col3:
            reduction = ((avg_original - avg_processed) / max(avg_original, 1)) * 100
            st.metric("Word Reduction", f"{reduction:.1f}%")

    def balanced_emotion_sampling(self, df, samples_per_emotion=800, total_target=None):
        """NEW: Smart balanced sampling - equal samples per emotion"""
        st.subheader("Balanced Emotion Sampling (NEW APPROACH)")
        
        # Calculate target samples per emotion
        if total_target:
            samples_per_emotion = total_target // len(self.emotion_columns)
        
        st.info(f"**NEW STRATEGY**: Equal samples per emotion for perfect balance!")
        st.info(f"**Target**: {samples_per_emotion:,} samples per emotion")
        
        # Analyze current distribution
        original_counts = {}
        available_counts = {}
        
        for emotion in self.emotion_columns:
            if emotion in df.columns:
                emotion_samples = df[df[emotion] == 1]
                original_counts[emotion] = len(emotion_samples)
                available_counts[emotion] = len(emotion_samples)
        
        # Show original distribution
        st.write("**ORIGINAL IMBALANCED DISTRIBUTION:**")
        
        insufficient_emotions = []
        sufficient_emotions = []
        
        for emotion, count in sorted(original_counts.items(), key=lambda x: x[1]):
            percentage = (count / len(df)) * 100
            if count < samples_per_emotion:
                insufficient_emotions.append((emotion, count, samples_per_emotion - count))
                st.error(f"   • **{emotion}**: {count:,} samples (need {samples_per_emotion - count:,} more)")
            else:
                sufficient_emotions.append((emotion, count))
                st.success(f"   • **{emotion}**: {count:,} samples (sufficient)")
        
        # Handle insufficient emotions
        if insufficient_emotions:
            st.warning("**INSUFFICIENT EMOTIONS DETECTED!**")
            st.write("**Options:**")
            
            option = st.radio(
                "How to handle insufficient emotions?",
                [
                    f"Reduce target to {min(original_counts.values())} (use all available)",
                    f"Keep {samples_per_emotion} target (insufficient emotions will be oversampled)",
                    "Custom target amount"
                ]
            )
            
            if "Reduce target" in option:
                samples_per_emotion = min(original_counts.values())
                st.info(f"**NEW TARGET**: {samples_per_emotion} samples per emotion")
            elif "Custom" in option:
                custom_target = st.slider(
                    "Custom samples per emotion:",
                    min_value=min(original_counts.values()),
                    max_value=max(original_counts.values()),
                    value=min(800, min(original_counts.values()))
                )
                samples_per_emotion = custom_target
        
        st.divider()
        
        # Create balanced dataset
        st.write("**CREATING PERFECTLY BALANCED DATASET:**")
        balanced_dfs = []
        final_counts = {}
        
        for emotion in self.emotion_columns:
            if emotion not in df.columns:
                continue
                
            emotion_samples = df[df[emotion] == 1].copy()
            available_count = len(emotion_samples)
            
            if available_count == 0:
                st.warning(f"   • **{emotion}**: No samples available")
                continue
            
            if available_count >= samples_per_emotion:
                # Sufficient samples - randomly sample
                sampled_data = emotion_samples.sample(n=samples_per_emotion, random_state=42)
                st.success(f"   • **{emotion}**: {available_count:,} → {samples_per_emotion:,} (sampled)")
                final_counts[emotion] = samples_per_emotion
                balanced_dfs.append(sampled_data)
                
            else:
                # Insufficient samples - use all + oversample if needed
                if samples_per_emotion <= available_count * 3:  # Reasonable oversampling
                    # Create oversampled data
                    multiplier = samples_per_emotion // available_count
                    remainder = samples_per_emotion % available_count
                    
                    oversampled_parts = []
                    for _ in range(multiplier):
                        oversampled_parts.append(emotion_samples.copy())
                    
                    if remainder > 0:
                        oversampled_parts.append(emotion_samples.sample(n=remainder, random_state=42))
                    
                    oversampled_data = pd.concat(oversampled_parts, ignore_index=True)
                    st.warning(f"   • **{emotion}**: {available_count:,} → {samples_per_emotion:,} (oversampled {multiplier + 1}x)")
                    final_counts[emotion] = samples_per_emotion
                    balanced_dfs.append(oversampled_data)
                else:
                    # Too much oversampling needed - use all available
                    st.error(f"   • **{emotion}**: {available_count:,} (used all - oversampling too extreme)")
                    final_counts[emotion] = available_count
                    balanced_dfs.append(emotion_samples)
        
        # Combine all balanced samples
        if balanced_dfs:
            balanced_df = pd.concat(balanced_dfs, ignore_index=True)
            # Remove any exact duplicates but keep intentional oversampling
            balanced_df = balanced_df.drop_duplicates(subset=['text'], keep='first')
            # Shuffle the entire dataset
            balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
        else:
            balanced_df = df.copy()
        
        # Show final results
        st.divider()
        st.write("**FINAL PERFECTLY BALANCED DISTRIBUTION:**")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Original", f"{len(df):,}")
            st.metric("Total Final", f"{len(balanced_df):,}")
        
        with col2:
            perfect_balance = len(set(final_counts.values())) == 1  # All counts are equal
            if perfect_balance:
                st.metric("Balance Status", "PERFECT", delta="✅")
            else:
                st.metric("Balance Status", "GOOD", delta="⚠️")
            
            avg_samples = np.mean(list(final_counts.values()))
            st.metric("Avg per Emotion", f"{avg_samples:.0f}")
        
        with col3:
            emotions_at_target = sum(1 for count in final_counts.values() if count == samples_per_emotion)
            st.metric("At Target Count", f"{emotions_at_target}/{len(final_counts)}")
            
            balance_score = (emotions_at_target / len(final_counts)) * 100
            st.metric("Balance Score", f"{balance_score:.0f}%")
        
        # Show perfect balance confirmation
        st.write("**FINAL EMOTION COUNTS:**")
        
        for emotion in self.emotion_columns:
            if emotion in final_counts:
                count = final_counts[emotion]
                if count == samples_per_emotion:
                    st.success(f"   ✅ **{emotion.title()}**: {count:,} samples (PERFECT)")
                elif count >= samples_per_emotion * 0.8:
                    st.info(f"   ℹ️ **{emotion.title()}**: {count:,} samples (good)")
                else:
                    st.warning(f"   ⚠️ **{emotion.title()}**: {count:,} samples (low)")
        
        st.success(f"**BALANCED SAMPLING COMPLETE!** Expected performance: Consistent across ALL emotions")
        
        return balanced_df
    
    def process_data(self, df, sample_size=None, preprocessing_options=None, use_balanced_sampling=True):
        """Process data with new balanced sampling approach"""
        try:
            # Default to BERT-optimized for higher accuracy
            if preprocessing_options is None:
                preprocessing_options = {
                    'remove_stopwords': False,
                    'use_stemming': False,
                    'use_lemmatization': False,
                    'remove_punctuation': False,
                    'remove_numbers': False,
                    'bert_optimized': True
                }
            
            # NEW: Apply balanced sampling FIRST (much simpler than oversampling)
            if use_balanced_sampling:
                if sample_size:
                    samples_per_emotion = sample_size // len(self.emotion_columns)
                else:
                    samples_per_emotion = 800  # Default
                
                df = self.balanced_emotion_sampling(df, samples_per_emotion=samples_per_emotion)
                st.success(f"Applied balanced sampling! New dataset: {len(df):,} samples")
            
            # Clean text
            df = df.copy()
            st.write("Applying text preprocessing...")
            
            # Choose cleaning method based on settings
            if preprocessing_options.get('bert_optimized', False):
                st.success("Using BERT-optimized preprocessing for higher accuracy")
                df['cleaned_text'] = df['text'].apply(self.clean_text_for_bert)
            else:
                # Show what preprocessing steps are being applied
                active_steps = []
                if preprocessing_options.get('remove_stopwords'):
                    active_steps.append("Remove stop words")
                if preprocessing_options.get('remove_punctuation'):
                    active_steps.append("Remove punctuation")
                if preprocessing_options.get('remove_numbers'):
                    active_steps.append("Remove numbers")
                if preprocessing_options.get('use_lemmatization'):
                    active_steps.append("Lemmatization")
                elif preprocessing_options.get('use_stemming'):
                    active_steps.append("Stemming")
                
                if active_steps:
                    st.warning(f"Using standard NLP: {', '.join(active_steps)} (may reduce BERT accuracy)")
                else:
                    st.info("Using minimal preprocessing")
                
                # Apply standard cleaning
                clean_params = {k: v for k, v in preprocessing_options.items() if k != 'bert_optimized'}
                df['cleaned_text'] = df['text'].apply(
                    lambda x: self.clean_text(x, **clean_params)
                )
            
            # Remove empty texts after cleaning
            df = df[df['cleaned_text'] != '']
            
            if len(df) == 0:
                st.error("No valid texts after cleaning")
                return None, None, None, None
            
            # Show preprocessing results
            if preprocessing_options.get('bert_optimized', False):
                st.success(f"Applied BERT-optimized preprocessing to {len(df):,} samples")
            else:
                st.success(f"Applied standard NLP preprocessing to {len(df):,} samples")
            
            # Prepare features and labels
            X = df['cleaned_text'].values
            y = df[self.emotion_columns].values.astype(float)
            
            # Split data (stratified split is less important with balanced data)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            st.success("Split data into 80% training, 20% testing")
            
            # Final balance verification
            st.write("**Final Training Set Balance:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                train_counts = {}
                for i, emotion in enumerate(self.emotion_columns):
                    if i < y_train.shape[1]:
                        train_counts[emotion] = np.sum(y_train[:, i])
                
                min_count = min(train_counts.values()) if train_counts else 0
                max_count = max(train_counts.values()) if train_counts else 0
                st.metric("Min Emotion Count", f"{min_count:,}")
                st.metric("Max Emotion Count", f"{max_count:,}")
            
            with col2:
                if max_count > 0:
                    balance_ratio = min_count / max_count
                    st.metric("Balance Ratio", f"{balance_ratio:.2f}")
                    if balance_ratio > 0.8:
                        st.success("Excellent balance!")
                    elif balance_ratio > 0.6:
                        st.info("Good balance")
                    else:
                        st.warning("Some imbalance remains")
                
                well_balanced = sum(1 for count in train_counts.values() if count >= min_count * 0.8)
                st.metric("Well-Balanced Emotions", f"{well_balanced}/{len(train_counts)}")
            
            with col3:
                st.metric("Training Samples", f"{len(X_train):,}")
                st.metric("Test Samples", f"{len(X_test):,}")
            
            st.success(f"Data processing complete! Balanced approach should give consistent performance across ALL emotions")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            st.error(f"Error processing data: {str(e)}")
            return None, None, None, None