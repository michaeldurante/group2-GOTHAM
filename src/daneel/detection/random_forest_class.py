import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)

import joblib

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class RandomForestDetection:
    def __init__(self, p_dictionary):

        self.csv_path = p_dictionary.get('dataset_path')
        self.n_bins = p_dictionary.get('n_bins', 1000)   
        self.max_depth = p_dictionary.get('max_depth')
        self.n_estimators = p_dictionary.get('n_estimators')
        self.use_scaler = p_dictionary.get('scalar', False)
        self.samples_per_class = p_dictionary.get('samples_per_class', 350)
        self.test_size = p_dictionary.get('test_size', 0.2)
        self.output_dir = p_dictionary.get('output_dir', "./results_rf")
        self.min_samples_leaf = p_dictionary.get('min_samples_leaf', 1)
        self.max_features = p_dictionary.get('max_features', 'sqrt')
        self.bootstrap = p_dictionary.get('bootstrap', True)
        self.class_weight = p_dictionary.get('class_weight', None)
        self.n_jobs = p_dictionary.get('n_jobs', -1)
        self.oob_score = p_dictionary.get('oob_score', True)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)


    def create_balanced_dataset(self):    
        X0 = self.X_train[self.y_train == 0]
        X1 = self.X_train[self.y_train == 1]

        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
            
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                X_result.append(X_aug)
            X_final = np.vstack(X_result)
            return X_final[:n_target]
    
        X0_bal = augment_to_target(X0, self.samples_per_class)
        X1_bal = augment_to_target(X1, self.samples_per_class)
        
        X_bal = np.vstack([X0_bal, X1_bal])
        y_bal = np.concatenate([np.zeros(self.samples_per_class), np.ones(self.samples_per_class)])
        
        idx = np.arange(len(X_bal))
        np.random.shuffle(idx)

        return X_bal[idx], y_bal[idx]

        
    def load_data(self):
        df = pd.read_csv(self.csv_path)
        
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]
        
        # print("Original distribution:")
        # print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        # if (y==0).sum() > 0:
        #     print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")
        
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_err_train, self.X_err_test, self.idx_train, self.idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=self.test_size,
            random_state=RANDOM_STATE,
            stratify=y
        )
        # print(f"Initial split - Train: {len(X_train)}, Test: {len(X_test)}")
        
        # Balance training set
        self.X_train, self.y_train = self.create_balanced_dataset()
        
        self.scaler = None
        if self.use_scaler:
            self.scaler = StandardScaler()
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
            # print(f"Train: mean={X_train.mean():.6f}, std={X_train.std():.6f}")
            # print(f"Test:  mean={X_test.mean():.6f}, std={X_test.std():.6f}")
        
        self.metadata_test = metadata.iloc[self.idx_test].reset_index(drop=True)
        # print(f"Final - X_train: {X_train.shape}, X_test: {X_test.shape}")
        # print(f"Train dist: 0={(y_train==0).sum()}, 1={(y_train==1).sum()}")
        
        # Return X_test copy (for optional inverse transform plotting if scaler is used)
        self.X_test_orig = self.X_test.copy()


    def build_random_forest(self):
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            bootstrap=self.bootstrap,
            class_weight=self.class_weight,
            random_state=RANDOM_STATE,
            n_jobs=self.n_jobs,
            oob_score=self.oob_score
        )


    def train_model(self):
        self.rf.fit(self.X_train, self.y_train)
        # if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
        #     print(f"OOB Score: {model.oob_score_:.4f}")


    def evaluate_with_optimal_threshold(self):

        self.proba = self.rf.predict_proba(self.X_test)[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y_test, self.proba)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        self.best_thresh = thresholds[best_idx]
        
        y_pred_default = (self.proba >= 0.5).astype(int)
        self.y_pred_best = (self.proba >= self.best_thresh).astype(int)
        
        auc = roc_auc_score(self.y_test, self.proba)
        acc_default = accuracy_score(self.y_test, y_pred_default)
        acc_best = accuracy_score(self.y_test, self.y_pred_best)

        self.roc_tuple = (fpr, tpr, thresholds)

        out_txt = self.output_dir / 'classifier_statistics.txt'
        with open(out_txt, 'w') as f:
            if hasattr(self.rf, 'oob_score_') and self.rf.oob_score_ is not None:
                print(f"OOB Score: {self.rf.oob_score_:.4f}", file=f)
            print("\n" + "="*70, file=f)
            print("THRESHOLD OPTIMIZATION & EVALUATION", file=f)
            print("="*70, file=f)
            
            print(f"Optimal threshold: {self.best_thresh:.4f} (default=0.5)", file=f)
            print(f"  At this threshold: TPR={tpr[best_idx]:.4f}, FPR={fpr[best_idx]:.4f}", file=f)
            print(f"AUC-ROC: {auc:.4f}", file=f)
            print(f"Accuracy @0.5: {acc_default:.4f} ({acc_default*100:.2f}%)", file=f)
            print(f"Accuracy @{self.best_thresh:.4f}: {acc_best:.4f} ({acc_best*100:.2f}%)", file=f)
            
            print("\nClassification report (optimal threshold):", file=f)
            print(classification_report(self.y_test, self.y_pred_best, target_names=['Non-Planet','Planet'], digits=4, zero_division=0), file=f)
        
            f.close()
        # print(f'Classifier statistics saved to {out_txt}')


    def plot_confusion_matrix_image(self, save_path='confusion_matrix_rf.png'):
        cm = confusion_matrix(self.y_test, self.y_pred_best)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        im = ax.imshow(cm, interpolation='nearest')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={self.best_thresh:.3f})')
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


    def plot_roc_curve(self, save_path='roc_curve_rf.png'):
        fpr, tpr, _ = roc_curve(self.y_test, self.proba)
        auc = roc_auc_score(self.y_test, self.proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(fpr, tpr, linewidth=2)
        ax.plot([0,1], [0,1], linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve (AUC={auc:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


    def plot_pr_curve(self, save_path='pr_curve_rf.png'):
        precision, recall, _ = precision_recall_curve(self.y_test, self.proba)
        ap = average_precision_score(self.y_test, self.proba)
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.plot(recall, precision, linewidth=2)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve (AP={ap:.4f})')
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


    def plot_probability_histograms(self, save_path='self.probability_hist_RF.png'):
        fig = plt.figure(figsize=(6, 5))
        ax = plt.gca()
        ax.hist(self.proba[self.y_test==0], bins=30, alpha=0.6, label='Non-Planet', density=True)
        ax.hist(self.proba[self.y_test==1], bins=30, alpha=0.6, label='Planet', density=True)
        ax.set_xlabel('Predicted Probability (class=1)')
        ax.set_ylabel('Density')
        ax.set_title('Predicted Probability Distributions by Class')
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close(fig)


    def main(self):
        self.load_data()
        self.build_random_forest()              # Build RF model
        self.train_model()                      # Train
        self.evaluate_with_optimal_threshold()  # Evaluate & choose best threshold

        # Visualizations
        self.plot_confusion_matrix_image(save_path=self.output_dir / 'confusion_matrix_rf.png')
        self.plot_roc_curve(save_path=self.output_dir / 'roc_curve_rf.png')
        self.plot_pr_curve(save_path=self.output_dir / 'pr_curve_rf.png')
        self.plot_probability_histograms(save_path=self.output_dir / 'probability_hist_RF.png')

        # Persist the model and the optimal threshold
        joblib.dump(self.rf, self.output_dir / 'tess_rf_model.joblib')
        np.save(self.output_dir / 'rf_optimal_threshold.npy', self.best_thresh)
        # print(f"Saved tess_rf_model.joblib and rf_optimal_threshold.npy in {self.output_dir}")
        print(f'All figures and ouputs saved in {self.output_dir}') 
