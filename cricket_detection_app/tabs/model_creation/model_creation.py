# enhanced_model_training.ipynb

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTENC, BorderlineSMOTE, SMOTE
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif, RFE
import matplotlib.pyplot as plt
import pickle

class ImprovedCricketModel:
    """
    Improved modeling pipeline with better handling of cricket object detection.
    """
    
    def __init__(self, file_path):
        self.file_path = file_path
        self.models = {}
        self.best_model = None
        self.scaler = None
        self.feature_selector = None
        self.pca = None
    
    def load_and_prepare_data(self):
        """
        Load data with better preprocessing.
        """
        df = pd.read_csv(self.file_path)
        #filter first 1000 rows for quick testing
        df = df.head(1000)
        print(f"Data loaded: {df.shape}")
        df_clean = df.dropna()
        
        #filter out rows with y=1
        df_clean = df_clean[df_clean['y'] != 1]
        
        X = df_clean.drop(['image', 'tile_i', 'tile_j', 'tile_number', 'y'], 
                         axis=1, errors='ignore')
        y = df_clean['y']
        
        
        
        print(f"Original data shape: {X.shape}")
        print(f"Class distribution:\n{y.value_counts()}")
        
        return X, y, df_clean
    
    def advanced_feature_selection(self, X, y, method='hybrid', n_features=None):
        """
        Advanced feature selection combining multiple methods.
        """
        if n_features is None:
            n_features = min(300, X.shape[1])
        
        if method == 'univariate':
            # Univariate feature selection
            selector = SelectKBest(f_classif, k=n_features)
            X_selected = selector.fit_transform(X, y)
            print(f"✓ Selected {n_features} features using univariate selection")
            self.feature_selector = selector
            return X_selected
        
        elif method == 'rfe':
            # Recursive Feature Elimination
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector = RFE(estimator, n_features_to_select=n_features, step=50)
            X_selected = selector.fit_transform(X, y)
            print(f"✓ Selected {n_features} features using RFE")
            self.feature_selector = selector
            return X_selected
        
        elif method == 'hybrid':
            # Combination: first univariate, then RFE
            # Step 1: Univariate to reduce to 500
            selector1 = SelectKBest(f_classif, k=min(500, X.shape[1]))
            X_temp = selector1.fit_transform(X, y)
            
            # Step 2: RFE to final n_features
            estimator = RandomForestClassifier(n_estimators=50, random_state=42)
            selector2 = RFE(estimator, n_features_to_select=n_features, step=20)
            X_selected = selector2.fit_transform(X_temp, y)
            
            print(f"✓ Selected {n_features} features using hybrid selection")
            self.feature_selector = (selector1, selector2)
            return X_selected
        
        else:
            return X
    
    def improved_sampling(self, X, y, strategy='borderline_smote'):
        """
        Improved sampling strategy for imbalanced data.
        """
        print(f"\nOriginal distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        
        # Random Undersampling of majority class (0)
        rus = RandomUnderSampler(
            sampling_strategy={0: len(y[y != 0])}, 
            random_state=42
        )
        X_rus, y_rus = rus.fit_resample(X, y)
        print(f"After undersampling: {dict(zip(*np.unique(y_rus, return_counts=True)))}")
        
        # Borderline SMOTE (better than regular SMOTE for overlapping classes)
        if strategy == 'borderline_smote':
            oversample = BorderlineSMOTE(random_state=42, kind='borderline-2')
        else:
            oversample = SMOTE(random_state=42)
        
        X_balanced, y_balanced = oversample.fit_resample(X_rus, y_rus)
        print(f"After SMOTE: {dict(zip(*np.unique(y_balanced, return_counts=True)))}")
        
        return X_balanced, y_balanced
    
    def create_ensemble_models(self):
        """
        Create ensemble of diverse models.
        """
        models = {
            # Base models
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                min_samples_split=10,
                min_samples_leaf=4,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            
            # 'GradientBoosting': GradientBoostingClassifier(
            #     n_estimators=150,
            #     max_depth=7,
            #     learning_rate=0.1,
            #     subsample=0.8,
            #     random_state=42
            # ),
            
            'SVC-RBF': SVC(
                kernel='rbf',
                C=10.0,
                gamma='scale',
                probability=True,
                class_weight='balanced',
                random_state=42
            )
            
            # 'MLP': MLPClassifier(
            #     hidden_layer_sizes=(128, 64, 32),
            #     activation='relu',
            #     solver='adam',
            #     alpha=0.001,
            #     batch_size=128,
            #     learning_rate='adaptive',
            #     max_iter=500,
            #     early_stopping=True,
            #     random_state=42
            # )
        }
        
        # Voting Classifier (Ensemble)
        # voting_clf = VotingClassifier(
        #     estimators=[
        #         ('rf', models['RandomForest']),
        #         ('gb', models['GradientBoosting']),
        #         ('svc', models['SVC-RBF'])
        #     ],
        #     voting='soft'
        # )
        
        # models['VotingEnsemble'] = voting_clf
        
        return models
    
    def train_with_cross_validation(self, X, y):
        """
        Train models with cross-validation for better generalization.
        """
        models = self.create_ensemble_models()
        
        # Stratified K-Fold
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"Training {model_name}...")
            print(f"{'='*70}")
            
            # Cross-validation scores
            cv_scores = cross_val_score(model, X, y, cv=skf, 
                                       scoring='f1_weighted', n_jobs=-1)
            
            print(f"CV F1 Scores: {cv_scores}")
            print(f"Mean CV F1: {np.mean(cv_scores):.4f} (+/- {np.std(cv_scores):.4f})")
            
            # Train on full training set
            model.fit(X, y)
            
            results[model_name] = {
                'model': model,
                'cv_mean': np.mean(cv_scores),
                'cv_std': np.std(cv_scores)
            }
        
        # Select best model based on CV score
        best_model_name = max(results, key=lambda x: results[x]['cv_mean'])
        self.best_model = results[best_model_name]['model']
        
        print(f"\n{'='*70}")
        print(f"Best Model: {best_model_name}")
        print(f"CV F1 Score: {results[best_model_name]['cv_mean']:.4f}")
        print(f"{'='*70}\n")
        
        return results, best_model_name
    
    def full_pipeline(self, use_pca=True, n_features=250):
        """
        Complete training pipeline.
        """
        # Load data
        X, y, df_meta = self.load_and_prepare_data()
        
        # Feature selection
        X_selected = self.advanced_feature_selection(X, y, method='hybrid', 
                                                     n_features=n_features)
        
        # Sampling
        X_balanced, y_balanced = self.improved_sampling(X_selected, y, 
                                                        strategy='borderline_smote')
        
        # Scaling
        self.scaler = RobustScaler()  # More robust to outliers than MinMaxScaler
        X_scaled = self.scaler.fit_transform(X_balanced)
        
        # Optional PCA
        if use_pca:
            self.pca = PCA(n_components=0.95, random_state=42)
            X_final = self.pca.fit_transform(X_scaled)
            print(f"PCA: {X_final.shape[1]} components (95% variance)")
        else:
            X_final = X_scaled
        
        # Train and test split
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X_final, y_balanced, test_size=0.2, 
            random_state=42, stratify=y_balanced
        )
        
        # Train with cross-validation
        results, best_model_name = self.train_with_cross_validation(X_train, y_train)
        
        # Evaluate on test set
        y_pred = self.best_model.predict(X_test)
        
        print("\n" + "="*70)
        print("TEST SET EVALUATION")
        print("="*70)
        # Classification report
        report_text = classification_report(y_test, y_pred)
        print(report_text)
        # Pretty visualization of classification report
        try:
            from sklearn.metrics import classification_report as cr
            import seaborn as sns
            import os
            report_dict = cr(y_test, y_pred, output_dict=True)
            # Build DataFrame with classes only
            cls_rows = {k: v for k, v in report_dict.items() if k not in ["accuracy", "macro avg", "weighted avg"]}
            df_rep = pd.DataFrame(cls_rows).T[["precision", "recall", "f1-score", "support"]]
            # Plot precision/recall/f1 bars per class
            fig2, ax2 = plt.subplots(figsize=(10, 6))
            df_plot = df_rep[["precision", "recall", "f1-score"]]
            df_plot.plot(kind="bar", ax=ax2, colormap="viridis")
            ax2.set_title("Classification Report (Precision/Recall/F1)", fontsize=14, fontweight="bold")
            ax2.set_xlabel("Class", fontsize=12)
            ax2.set_ylabel("Score", fontsize=12)
            ax2.legend(loc="upper right")
            ax2.grid(axis="y", alpha=0.2)
            plt.tight_layout()
            # Save next to models
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
            models_dir = os.path.abspath(models_dir)
            os.makedirs(models_dir, exist_ok=True)
            rep_path = os.path.join(models_dir, 'classification_report.png')
            fig2.savefig(rep_path, dpi=160)
            print(f"\n✓ Classification report chart saved to: {rep_path}")
            plt.close(fig2)
        except Exception as rep_err:
            print(f"Could not render/save classification report chart: {rep_err}")
        
        # Confusion matrix (styled visualization)
        cm = confusion_matrix(y_test, y_pred)
        print("\nConfusion Matrix:")
        print(cm)

        try:
            import seaborn as sns
            import os
            # Prepare nice labels
            labels = sorted(np.unique(y_test))
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(
                cm,
                annot=True,
                fmt="d",
                cmap="YlGnBu",
                linewidths=0.5,
                linecolor="white",
                cbar=True,
                xticklabels=labels,
                yticklabels=labels,
                ax=ax
            )
            ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
            ax.set_xlabel("Predicted", fontsize=12)
            ax.set_ylabel("Actual", fontsize=12)
            plt.tight_layout()

            # Save under models directory alongside pickles
            models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
            models_dir = os.path.abspath(models_dir)
            os.makedirs(models_dir, exist_ok=True)
            out_path = os.path.join(models_dir, 'confusion_matrix.png')
            fig.savefig(out_path, dpi=160)
            print(f"\n✓ Confusion matrix saved to: {out_path}")
            plt.close(fig)
        except Exception as viz_err:
            print(f"Could not render/save confusion matrix heatmap: {viz_err}")
        
        # Save models
        self.save_models(best_model_name)
        
        return results
    
    def save_models(self, best_model_name):
        """
        Save trained models and preprocessors.
        """
        import os
        # Resolve a project-local models directory and ensure it exists
        models_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models')
        models_dir = os.path.abspath(models_dir)
        os.makedirs(models_dir, exist_ok=True)

        best_model_path = os.path.join(models_dir, 'best_model.pkl')
        with open(best_model_path, 'wb') as f:
            pickle.dump(self.best_model, f)

        scaler_path = os.path.join(models_dir, 'scaler.pkl')
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)

        if self.pca:
            pca_path = os.path.join(models_dir, 'pca_model.pkl')
            with open(pca_path, 'wb') as f:
                pickle.dump(self.pca, f)

        if self.feature_selector:
            feature_selector_path = os.path.join(models_dir, 'feature_selector.pkl')
            with open(feature_selector_path, 'wb') as f:
                pickle.dump(self.feature_selector, f)
        
        print(f"\n✓ Models saved to {models_dir} (Best: {best_model_name})")


def run_full_pipeline(file_path: str, use_pca: bool = True, n_features: int = 250):
    """Convenience wrapper to run the full pipeline without instantiating externally.
    Returns the training `results` dict and the best model name.
    """
    trainer = ImprovedCricketModel(file_path)
    results = trainer.full_pipeline(use_pca=use_pca, n_features=n_features)
    # best model name is printed inside; return results for external usage
    return results

def prepare_data(file_path: str):
    """Load and prepare data using the class logic, returning X, y, df_meta."""
    trainer = ImprovedCricketModel(file_path)
    return trainer.load_and_prepare_data()

def train_models(X, y):
    """Train models (cross-validation and fit) given precomputed X, y arrays.
    Returns (results, best_model_name, trained_trainer).
    """
    # Minimal trainer to reuse saving and attributes
    dummy_file = '<in-memory>'
    trainer = ImprovedCricketModel(dummy_file)
    results, best_model_name = trainer.train_with_cross_validation(X, y)
    return results, best_model_name, trainer

if __name__ == "__main__":
    # Example usage guarded to avoid auto-execution when imported
    default_file = '../dataset/enhanced_test_data/enhanced_features_consolidation.csv'
    try:
        run_full_pipeline(default_file, use_pca=True, n_features=250)
    except Exception as e:
        print(f"Pipeline execution failed: {e}")