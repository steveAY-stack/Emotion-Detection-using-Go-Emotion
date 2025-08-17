import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, hamming_loss, roc_auc_score, roc_curve
import warnings

class SimpleModelEvaluator:
    def __init__(self):
        self.emotion_labels = [
            'admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring',
            'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval',
            'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief',
            'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization',
            'relief', 'remorse', 'sadness', 'surprise', 'neutral'
        ]
    
    def evaluate_models(self, classifiers, X_test, y_test):
        """Evaluate models with proper multi-label metrics on balanced data"""
        results = {}
        
        try:
            # Evaluate Naive Bayes
            if classifiers.nb_classifier:
                nb_metrics = self._evaluate_single_model(
                    classifiers.nb_classifier, X_test, y_test, "Naive Bayes", classifiers.pca
                )
                if nb_metrics:
                    results['naive_bayes'] = nb_metrics
            
            # Evaluate Random Forest
            if classifiers.rf_classifier:
                rf_metrics = self._evaluate_single_model(
                    classifiers.rf_classifier, X_test, y_test, "Random Forest", None
                )
                if rf_metrics:
                    results['random_forest'] = rf_metrics
            
            # Show detailed comparison
            if results:
                self._show_detailed_comparison(results, y_test)
            
            return results
            
        except Exception as e:
            st.error(f"Error evaluating models: {str(e)}")
            return None
    
    def _safe_roc_auc_calculation(self, y_true, y_pred_proba, average='macro'):
        """Robust ROC-AUC calculation that handles edge cases"""
        try:
            # Check if we have valid data
            if y_true.size == 0 or y_pred_proba.size == 0:
                st.warning("Empty data for ROC-AUC calculation")
                return 0.5
            
            # Check for emotions with only one class (all 0s or all 1s)
            valid_emotions = []
            for i in range(y_true.shape[1]):
                emotion_true = y_true[:, i]
                unique_values = np.unique(emotion_true)
                
                if len(unique_values) > 1:  # Has both positive and negative samples
                    valid_emotions.append(i)
                else:
                    # Skip emotions with only one class
                    continue
            
            if len(valid_emotions) == 0:
                st.warning("No emotions with both positive and negative samples for ROC-AUC")
                return 0.5
            
            # Calculate ROC-AUC only for valid emotions
            y_true_valid = y_true[:, valid_emotions]
            y_pred_proba_valid = y_pred_proba[:, valid_emotions]
            
            # Try macro average first
            try:
                roc_auc_macro = roc_auc_score(y_true_valid, y_pred_proba_valid, average='macro', multi_class='ovr')
                return float(roc_auc_macro)
            except ValueError as e:
                if "multi_class" in str(e):
                    # Fallback: calculate per-emotion and average
                    auc_scores = []
                    for i in range(y_true_valid.shape[1]):
                        try:
                            auc = roc_auc_score(y_true_valid[:, i], y_pred_proba_valid[:, i])
                            auc_scores.append(auc)
                        except ValueError:
                            continue
                    
                    if auc_scores:
                        return float(np.mean(auc_scores))
                    else:
                        return 0.5
                else:
                    raise e
            
        except Exception as e:
            st.warning(f"ROC-AUC calculation failed: {str(e)}. Using fallback value.")
            return 0.5  # Return neutral performance as fallback
    
    def _evaluate_single_model(self, model, X_test, y_test, model_name, pca=None):
        """Evaluate model with robust multi-label metrics on balanced test data"""
        try:
            # Apply PCA if needed (for Naive Bayes)
            X_test_processed = X_test
            if pca is not None:
                X_test_processed = pca.transform(X_test)
            
            # Get probability predictions
            y_pred_proba = model.predict_proba(X_test_processed)
            
            # Handle MultiOutputClassifier probability format
            if isinstance(y_pred_proba, list):
                # Convert list of arrays to matrix
                proba_matrix = np.zeros((len(X_test_processed), len(self.emotion_labels)))
                for i, emotion_proba in enumerate(y_pred_proba):
                    if i < len(self.emotion_labels):
                        if emotion_proba.shape[1] == 2:  # Binary classifier
                            proba_matrix[:, i] = emotion_proba[:, 1]  # Positive class
                        else:
                            proba_matrix[:, i] = emotion_proba[:, 0]  # Single value
                y_pred_proba = proba_matrix
            
            # Convert probabilities to binary predictions with threshold
            threshold = 0.5
            y_pred = (y_pred_proba >= threshold).astype(int)
            
            # Ensure dimensions match
            if y_pred.shape[1] > y_test.shape[1]:
                y_pred = y_pred[:, :y_test.shape[1]]
                y_pred_proba = y_pred_proba[:, :y_test.shape[1]]
            elif y_pred.shape[1] < y_test.shape[1]:
                # Pad with zeros
                padding = np.zeros((y_pred.shape[0], y_test.shape[1] - y_pred.shape[1]))
                y_pred = np.hstack([y_pred, padding])
                padding_proba = np.zeros((y_pred_proba.shape[0], y_test.shape[1] - y_pred_proba.shape[1]))
                y_pred_proba = np.hstack([y_pred_proba, padding_proba])
            
            # Calculate proper multi-label metrics
            metrics = {}
            
            # 1. SUBSET ACCURACY (exact match) - This is the correct "accuracy" for multi-label
            subset_accuracy = np.mean(np.all(y_test == y_pred, axis=1))
            metrics['subset_accuracy'] = float(subset_accuracy)
            
            # 2. HAMMING LOSS (element-wise accuracy)
            hamming_loss_score = hamming_loss(y_test, y_pred)
            hamming_accuracy = 1 - hamming_loss_score  # Convert to accuracy
            metrics['hamming_accuracy'] = float(hamming_accuracy)
            
            # 3. Use hamming accuracy as main "accuracy" metric
            metrics['accuracy'] = float(hamming_accuracy)
            
            # 4. Standard multi-label metrics with zero_division handling
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                metrics['macro_f1'] = float(f1_score(y_test, y_pred, average='macro', zero_division=0))
                metrics['weighted_f1'] = float(f1_score(y_test, y_pred, average='weighted', zero_division=0))
                metrics['macro_precision'] = float(precision_score(y_test, y_pred, average='macro', zero_division=0))
                metrics['macro_recall'] = float(recall_score(y_test, y_pred, average='macro', zero_division=0))
            
            # 5. Robust ROC-AUC for multi-label classification
            roc_auc_macro = self._safe_roc_auc_calculation(y_test, y_pred_proba, average='macro')
            metrics['roc_auc_macro'] = roc_auc_macro
            metrics['roc_auc'] = roc_auc_macro  # Use macro as main ROC-AUC metric
            
            # Try weighted ROC-AUC as well
            roc_auc_weighted = self._safe_roc_auc_calculation(y_test, y_pred_proba, average='weighted')
            metrics['roc_auc_weighted'] = roc_auc_weighted
            
            # 6. Per-emotion performance (balanced data analysis)
            emotion_performance = {}
            emotion_f1_scores = []
            emotion_balanced_scores = []
            
            for i, emotion in enumerate(self.emotion_labels):
                if i < y_test.shape[1]:
                    y_true_emotion = y_test[:, i]
                    y_pred_emotion = y_pred[:, i]
                    y_proba_emotion = y_pred_proba[:, i]
                    
                    # Only calculate if emotion exists in test set and has both classes
                    unique_values = np.unique(y_true_emotion)
                    if len(unique_values) > 1:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore")
                            emotion_f1 = f1_score(y_true_emotion, y_pred_emotion, zero_division=0)
                            emotion_precision = precision_score(y_true_emotion, y_pred_emotion, zero_division=0)
                            emotion_recall = recall_score(y_true_emotion, y_pred_emotion, zero_division=0)
                        
                        # Calculate per-emotion ROC-AUC safely
                        try:
                            emotion_roc_auc = roc_auc_score(y_true_emotion, y_proba_emotion)
                        except ValueError:
                            emotion_roc_auc = 0.5  # Neutral performance for problematic emotions
                        
                        emotion_performance[emotion] = {
                            'f1': float(emotion_f1),
                            'precision': float(emotion_precision),
                            'recall': float(emotion_recall),
                            'roc_auc': float(emotion_roc_auc),
                            'support': int(np.sum(y_true_emotion))
                        }
                        
                        emotion_f1_scores.append(emotion_f1)
                        
                        # For balanced data, check if performance is consistent
                        if emotion_f1 > 0.4:  # Good performance threshold
                            emotion_balanced_scores.append(emotion_f1)
                        
                    else:
                        # Single class emotion - add with neutral metrics
                        emotion_performance[emotion] = {
                            'f1': 0.0,
                            'precision': 0.0,
                            'recall': 0.0,
                            'roc_auc': 0.5,
                            'support': int(np.sum(y_true_emotion)),
                            'note': 'Single class - metrics not meaningful'
                        }
            
            metrics['emotion_performance'] = emotion_performance
            
            # 7. Balanced data performance assessment
            if emotion_f1_scores:
                avg_emotion_f1 = np.mean(emotion_f1_scores)
                std_emotion_f1 = np.std(emotion_f1_scores)
                metrics['avg_emotion_f1'] = float(avg_emotion_f1)
                metrics['std_emotion_f1'] = float(std_emotion_f1)
                
                # Check consistency (balanced data should have low std deviation)
                if std_emotion_f1 < 0.15:
                    metrics['balance_assessment'] = "Excellent - Very consistent performance across emotions"
                elif std_emotion_f1 < 0.25:
                    metrics['balance_assessment'] = "Good - Reasonably consistent performance"
                elif std_emotion_f1 < 0.35:
                    metrics['balance_assessment'] = "Fair - Some variation in emotion performance"
                else:
                    metrics['balance_assessment'] = "Poor - High variation despite balanced training"
                
                # Count well-performing emotions
                good_emotions = sum(1 for f1 in emotion_f1_scores if f1 > 0.4)
                total_emotions = len(emotion_f1_scores)
                metrics['good_emotion_ratio'] = float(good_emotions / max(total_emotions, 1))
                
                if good_emotions / total_emotions > 0.8:
                    metrics['consistency_success'] = "Excellent - Most emotions performing well"
                elif good_emotions / total_emotions > 0.6:
                    metrics['consistency_success'] = "Good - Majority of emotions performing well"
                elif good_emotions / total_emotions > 0.4:
                    metrics['consistency_success'] = "Fair - Some emotions performing well"
                else:
                    metrics['consistency_success'] = "Poor - Few emotions performing well"
            
            # 8. Confidence metrics
            if y_pred_proba.size > 0:
                avg_confidence = np.mean(np.max(y_pred_proba, axis=1))
                metrics['avg_confidence'] = float(avg_confidence)
                
                # High confidence predictions
                high_confidence = np.mean(np.max(y_pred_proba, axis=1) > 0.7)
                metrics['high_confidence_ratio'] = float(high_confidence)
            
            # 9. Performance quality assessment for balanced data
            quality_score = self._assess_balanced_performance_quality(metrics)
            metrics['quality_assessment'] = quality_score
            
            # Show key metrics including balanced data effectiveness
            st.info(f"{model_name} evaluation completed on balanced test data")
            
            # Show balanced data effectiveness
            if 'balance_assessment' in metrics:
                if "Excellent" in metrics['balance_assessment']:
                    st.success(f"**Balanced Data Impact**: {metrics['balance_assessment']}")
                elif "Good" in metrics['balance_assessment']:
                    st.success(f"**Balanced Data Impact**: {metrics['balance_assessment']}")
                elif "Fair" in metrics['balance_assessment']:
                    st.warning(f"**Balanced Data Impact**: {metrics['balance_assessment']}")
                else:
                    st.error(f"**Balanced Data Impact**: {metrics['balance_assessment']}")
            
            return metrics
            
        except Exception as e:
            st.error(f"Error evaluating {model_name}: {str(e)}")
            st.exception(e)
            return None
    
    def _assess_balanced_performance_quality(self, metrics):
        """Assess overall model performance quality on balanced data"""
        hamming_acc = metrics.get('hamming_accuracy', 0)
        roc_auc = metrics.get('roc_auc', 0.5)
        f1_score = metrics.get('macro_f1', 0)
        consistency = 1 - metrics.get('std_emotion_f1', 1)  # Lower std = higher consistency
        
        # Weighted score emphasizing consistency for balanced data
        quality_score = (hamming_acc * 0.3) + (roc_auc * 0.25) + (f1_score * 0.25) + (consistency * 0.2)
        
        if quality_score >= 0.8:
            return "Excellent"
        elif quality_score >= 0.7:
            return "Good"
        elif quality_score >= 0.6:
            return "Fair"
        else:
            return "Poor"
    
    def _show_detailed_comparison(self, results, y_test):
        """Show detailed model comparison with balanced data metrics"""
        st.subheader("Model Performance Comparison (Balanced Test Data)")
        
        # Overall Performance Summary with consistency metrics
        st.write("**Complete Multi-Label Performance Metrics on Balanced Data:**")
        comparison_data = []
        
        for model_name, metrics in results.items():
            model_display_name = model_name.replace('_', ' ').title()
            
            # Use metrics relevant for balanced data
            subset_accuracy = metrics.get('subset_accuracy', 0) * 100
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            macro_f1 = metrics.get('macro_f1', 0) * 100
            macro_precision = metrics.get('macro_precision', 0) * 100
            macro_recall = metrics.get('macro_recall', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            avg_emotion_f1 = metrics.get('avg_emotion_f1', 0) * 100
            std_emotion_f1 = metrics.get('std_emotion_f1', 0) * 100
            quality = metrics.get('quality_assessment', 'Unknown')
            balance_status = metrics.get('balance_assessment', 'Unknown')
            
            comparison_data.append({
                'Model': model_display_name,
                'Hamming Accuracy': f"{hamming_accuracy:.1f}%",
                'Precision': f"{macro_precision:.1f}%",
                'Recall': f"{macro_recall:.1f}%",
                'F-Measure': f"{macro_f1:.1f}%",
                'ROC-AUC': f"{roc_auc:.1f}%",
                'Avg Emotion F1': f"{avg_emotion_f1:.1f}%",
                'F1 Std Dev': f"{std_emotion_f1:.1f}%",
                'Subset Accuracy': f"{subset_accuracy:.1f}%",
                'Quality': quality,
                'Consistency': balance_status.split(' - ')[0]
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Highlight best performing model and consistency
        if len(comparison_data) > 1:
            best_hamming = max(comparison_data, key=lambda x: float(x['Hamming Accuracy'].rstrip('%')))
            best_roc = max(comparison_data, key=lambda x: float(x['ROC-AUC'].rstrip('%')))
            best_f1 = max(comparison_data, key=lambda x: float(x['F-Measure'].rstrip('%')))
            most_consistent = min(comparison_data, key=lambda x: float(x['F1 Std Dev'].replace('±', '').replace('%', '').rstrip('%')))

            st.write("**Best Performance on Balanced Data:**")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Best Hamming Accuracy", f"{best_hamming['Model']}", 
                         delta=f"{best_hamming['Hamming Accuracy']}")
            with col2:
                st.metric("Best ROC-AUC", f"{best_roc['Model']}", 
                         delta=f"{best_roc['ROC-AUC']}")
            with col3:
                st.metric("Best F-Measure", f"{best_f1['Model']}", 
                         delta=f"{best_f1['F-Measure']}")
            with col4:
                st.metric("Most Consistent", f"{most_consistent['Model']}", 
                         delta=f"±{most_consistent['F1 Std Dev']}")
        
        # Show balanced data effectiveness summary
        st.write("**Balanced Data Effectiveness Summary:**")
        for model_name, metrics in results.items():
            model_display = model_name.replace('_', ' ').title()
            avg_f1 = metrics.get('avg_emotion_f1', 0) * 100
            std_f1 = metrics.get('std_emotion_f1', 0) * 100
            balance_status = metrics.get('balance_assessment', 'Unknown')
            
            if "Excellent" in balance_status:
                st.success(f"**{model_display}**: {balance_status} (Avg F1: {avg_f1:.1f}%, Std: ±{std_f1:.1f}%)")
            elif "Good" in balance_status:
                st.success(f"**{model_display}**: {balance_status} (Avg F1: {avg_f1:.1f}%, Std: ±{std_f1:.1f}%)")
            elif "Fair" in balance_status:
                st.warning(f"**{model_display}**: {balance_status} (Avg F1: {avg_f1:.1f}%, Std: ±{std_f1:.1f}%)")
            else:
                st.info(f"**{model_display}**: {balance_status} (Avg F1: {avg_f1:.1f}%, Std: ±{std_f1:.1f}%)")
        
        # Enhanced explanation of metrics for balanced data
        with st.expander("Understanding Metrics on Balanced Data"):
            st.write("""
            **Primary Performance Metrics for Balanced Data:**
            
            **Hamming Accuracy**: Average accuracy across all emotion labels (Main metric)
            - More forgiving - measures per-emotion accuracy
            - Expected: 65-85%+ for balanced data
            
            **Precision**: How many predicted positive emotions were actually correct
            - High precision = few false positives
            
            **Recall**: How many actual positive emotions were correctly identified  
            - High recall = few false negatives
            
            **F-Measure (F1-Score)**: Harmonic mean of precision and recall
            - Balances precision and recall
            - Expected: 50-70%+ with balanced data
            
            **ROC-AUC**: Area Under the Receiver Operating Characteristic curve
            - Measures ability to distinguish between classes
            - Range: 0.5 (random) to 1.0 (perfect)
            - Expected: 75-90%+ with balanced training
            
            **Balanced Data Specific Metrics:**
            
            **Average Emotion F1**: Mean F1-score across all emotions
            - Shows overall emotion detection capability
            - Expected: 50-70%+ with balanced sampling
            
            **F1 Standard Deviation**: Consistency of performance across emotions
            - Lower values = more consistent performance
            - <15% = Excellent consistency
            - 15-25% = Good consistency
            - >25% = Poor consistency (balanced data should prevent this)
            
            **Consistency Assessment**: Overall assessment of performance uniformity
            - Shows if balanced sampling achieved its goal of equal attention to all emotions
            
            **Why This Matters:**
            Balanced sampling should eliminate the "rare emotion problem" and give consistent 
            performance across all 27 emotions. The consistency metrics show if this worked.
            """)
        
        # Show which metrics to focus on for balanced data
        st.success("Model evaluation completed on perfectly balanced test data!")
    
    def create_roc_curves(self, classifiers, X_test, y_test):
        """Create ROC-AUC visualization for diverse emotions from balanced data"""
        try:
            # Select 6 emotions representing different categories for visualization
            selected_emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'neutral']
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=3,
                subplot_titles=[emotion.title() for emotion in selected_emotions],
                specs=[[{"secondary_y": False}]*3]*2
            )
            
            colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
            
            for idx, emotion in enumerate(selected_emotions):
                row = (idx // 3) + 1
                col = (idx % 3) + 1
                
                if emotion not in self.emotion_labels:
                    continue
                    
                emotion_idx = self.emotion_labels.index(emotion)
                y_true = y_test[:, emotion_idx]
                
                # Skip if only one class
                if len(np.unique(y_true)) < 2:
                    continue
                
                # Get predictions from both models
                models_data = []
                
                if classifiers.rf_classifier:
                    rf_proba = classifiers.rf_classifier.predict_proba(X_test)
                    if isinstance(rf_proba, list):
                        rf_emotion_proba = rf_proba[emotion_idx][:, 1] if rf_proba[emotion_idx].shape[1] == 2 else rf_proba[emotion_idx][:, 0]
                    else:
                        rf_emotion_proba = rf_proba[:, emotion_idx]
                    models_data.append(('Random Forest', rf_emotion_proba, colors[0]))
                
                if classifiers.nb_classifier and classifiers.pca:
                    X_test_pca = classifiers.pca.transform(X_test)
                    nb_proba = classifiers.nb_classifier.predict_proba(X_test_pca)
                    if isinstance(nb_proba, list):
                        nb_emotion_proba = nb_proba[emotion_idx][:, 1] if nb_proba[emotion_idx].shape[1] == 2 else nb_proba[emotion_idx][:, 0]
                    else:
                        nb_emotion_proba = nb_proba[:, emotion_idx]
                    models_data.append(('Naive Bayes', nb_emotion_proba, colors[1]))
                
                # Plot ROC curves for each model
                for model_name, proba, color in models_data:
                    try:
                        fpr, tpr, _ = roc_curve(y_true, proba)
                        auc_score = roc_auc_score(y_true, proba)
                        
                        display_name = f'{model_name} (AUC={auc_score:.3f})'
                        
                        fig.add_trace(
                            go.Scatter(
                                x=fpr, y=tpr,
                                mode='lines',
                                name=display_name,
                                line=dict(color=color, width=2),
                                showlegend=(idx == 0)  # Only show legend for first subplot
                            ),
                            row=row, col=col
                        )
                    except Exception:
                        continue
                
                # Add diagonal line (random classifier)
                fig.add_trace(
                    go.Scatter(
                        x=[0, 1], y=[0, 1],
                        mode='lines',
                        line=dict(color='gray', dash='dash', width=1),
                        showlegend=False
                    ),
                    row=row, col=col
                )
            
            # Update layout
            fig.update_layout(
                title="ROC-AUC Curves: Representative Emotions from Balanced Data",
                height=600,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            # Update axes
            for i in range(1, 7):
                row = ((i-1) // 3) + 1
                col = ((i-1) % 3) + 1
                fig.update_xaxes(title_text="False Positive Rate", row=row, col=col)
                fig.update_yaxes(title_text="True Positive Rate", row=row, col=col)
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating ROC curves: {str(e)}")
            return None
    
    def display_performance_summary(self, results):
        """Display performance summary with balanced data metrics"""
        st.subheader("Performance Summary - Balanced Data Results")
        
        if not results:
            st.error("No results to display")
            return
        
        summary_data = []
        
        for model_name, metrics in results.items():
            # Use metrics appropriate for balanced data evaluation
            hamming_accuracy = metrics.get('hamming_accuracy', 0) * 100
            macro_precision = metrics.get('macro_precision', 0) * 100
            macro_recall = metrics.get('macro_recall', 0) * 100
            macro_f1 = metrics.get('macro_f1', 0) * 100
            roc_auc = metrics.get('roc_auc', 0) * 100
            avg_emotion_f1 = metrics.get('avg_emotion_f1', 0) * 100
            std_emotion_f1 = metrics.get('std_emotion_f1', 0) * 100
            quality = metrics.get('quality_assessment', 'Unknown')
            consistency = metrics.get('balance_assessment', 'Unknown').split(' - ')[0]
            
            summary_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Accuracy': f"{hamming_accuracy:.1f}%",
                'Precision': f"{macro_precision:.1f}%", 
                'Recall': f"{macro_recall:.1f}%",
                'F-Measure': f"{macro_f1:.1f}%",
                'ROC-AUC': f"{roc_auc:.1f}%",
                'Avg Emotion F1': f"{avg_emotion_f1:.1f}%",
                'F1 Consistency': f"{std_emotion_f1:.1f}%",
                'Quality': quality,
                'Balance Success': consistency
            })
        
        # Display as table
        summary_df = pd.DataFrame(summary_data)
        st.dataframe(summary_df, use_container_width=True, hide_index=True)
        
        # Show best model and consistency analysis
        if len(summary_data) > 1:
            best_model = max(summary_data, key=lambda x: float(x['Accuracy'].replace('±', '').replace('%', '').strip()))
            most_consistent = min(summary_data, key=lambda x: float(x['F1 Consistency'].replace('±', '').replace('%', '').strip()))
            
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"**Best Overall Model**: {best_model['Model']} ({best_model['Accuracy']} accuracy)")
            with col2:
                st.success(f"**Most Consistent**: {most_consistent['Model']} ({most_consistent['F1 Consistency']} variation)")
        
        # Show balanced sampling effectiveness summary
        st.write("**Balanced Sampling Effectiveness:**")
        for data in summary_data:
            consistency_val = float(data['F1 Consistency'].rstrip('±%'))
            if consistency_val < 15:
                st.success(f"✓ **{data['Model']}**: Excellent consistency ({data['F1 Consistency']} variation)")
            elif consistency_val < 25:
                st.success(f"✓ **{data['Model']}**: Good consistency ({data['F1 Consistency']} variation)")
            elif consistency_val < 35:
                st.warning(f"⚠ **{data['Model']}**: Fair consistency ({data['F1 Consistency']} variation)")
            else:
                st.error(f"✗ **{data['Model']}**: Poor consistency ({data['F1 Consistency']} variation)")
        
        return summary_data
    
    def explain_metrics(self):
        """Explain key metrics for balanced data evaluation"""
        with st.expander("Understanding Balanced Data Performance Metrics"):
            st.write("""
            **Key Performance Metrics for Balanced Data:**
            
            **Precision**: Accuracy of positive predictions
            - How many predicted emotions were actually correct
            
            **Recall**: Coverage of actual positives  
            - How many actual emotions were correctly identified
            
            **F-Measure**: Balance between precision and recall
            - Harmonic mean of precision and recall
            
            **ROC-AUC**: Model's discriminative ability
            - Higher values indicate better separation between classes
            - Range: 0.5 (random) to 1.0 (perfect)
            
            **Hamming Accuracy**: Average accuracy across all emotion labels
            - More appropriate for multi-label classification
            
            **BALANCED DATA SPECIFIC METRICS:**
            
            **Average Emotion F1**: Shows overall emotion detection capability
            - With balanced data, this should be high and consistent
            - Target: 50-70%+ for good balanced performance
            
            **F1 Consistency (Standard Deviation)**: Shows performance uniformity
            - Lower values = more consistent performance across emotions
            - <15% = Excellent (balanced sampling worked perfectly)
            - 15-25% = Good (balanced sampling mostly worked)
            - >25% = Poor (balanced sampling didn't achieve consistency)
            
            **Balance Success**: Overall assessment of sampling effectiveness
            - Shows if balanced sampling achieved equal attention to all emotions
            
            **Why This Matters:**
            Balanced sampling should eliminate performance disparities between emotions.
            The consistency metrics show if our equal sampling strategy worked!
            """)
    
    def get_model_recommendations(self, results):
        """Provide recommendations based on balanced data performance"""
        recommendations = []
        
        if len(results) > 1:
            # Compare models
            best_acc = max(results.items(), key=lambda x: x[1].get('hamming_accuracy', 0))
            best_auc = max(results.items(), key=lambda x: x[1].get('roc_auc', 0))
            most_consistent = min(results.items(), key=lambda x: x[1].get('std_emotion_f1', 1))
            
            recommendations.append("**Model Performance Summary on Balanced Data:**")
            recommendations.append(f"• Best Overall Accuracy: {best_acc[0].replace('_', ' ').title()}")
            recommendations.append(f"• Best ROC-AUC: {best_auc[0].replace('_', ' ').title()}")
            recommendations.append(f"• Most Consistent Performance: {most_consistent[0].replace('_', ' ').title()}")
            
            # Assess balanced sampling success
            recommendations.append("")
            recommendations.append("**Balanced Sampling Effectiveness:**")
            for model_name, metrics in results.items():
                std_f1 = metrics.get('std_emotion_f1', 1) * 100
                if std_f1 < 15:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Excellent consistency (±{std_f1:.1f}% F1 variation)")
                elif std_f1 < 25:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Good consistency (±{std_f1:.1f}% F1 variation)")
                else:
                    recommendations.append(f"• {model_name.replace('_', ' ').title()}: Needs improvement (±{std_f1:.1f}% F1 variation)")
            
        else:
            recommendations.append("**Model evaluation completed successfully on balanced data.**")
        
        return recommendations