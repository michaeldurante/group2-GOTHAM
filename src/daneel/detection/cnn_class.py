import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler

import tensorflow as tf
import keras
from keras import layers, models, regularizers
# from tensorflow.keras import layers, models, regularizers
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras import backend as K

try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if gpus:
        print(f"Enabled memory growth for {len(gpus)} GPU(s).")
except Exception as e:
    print("GPU setup note:", e)

print("TF version:", tf.__version__)

# reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

class ConvolutionalNeuralNetwork:
    def __init__(self, p_dictionary):        
        self.alpha = p_dictionary.get('focal_alpha', 0.75)
        self.gamma = p_dictionary.get('focal_gamma', 2.5)
        self.epochs = p_dictionary.get('epochs', 100)
        self.samples_per_class = p_dictionary.get('samples_per_class', 400)
        self.n_bins = p_dictionary.get('n_bins', 1000)
        self.dataset_path = p_dictionary.get('dataset_path')
        self.n_samples = p_dictionary.get('n_samples', 8)
        self.output_path = p_dictionary.get('ouput_path', 'sample_lightcurves_predictions.png')
        self.output_dir = p_dictionary.get('output_dir', './results_cnn')

        try:
            os.makedirs(self.output_dir)
        except FileExistsError:
            # directory already exists
            pass
        

    def focal_loss(self, y_true, y_pred):
        epsilon = K.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
        
        pt = tf.where(tf.equal(y_true, 1), y_pred, 1 - y_pred)
        alpha_factor = tf.where(tf.equal(y_true, 1), self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * tf.math.pow(1 - pt, self.gamma)
        bce = -tf.math.log(pt)
        return tf.reduce_mean(focal_weight * bce)


    def create_balanced_dataset(self):
        """Create a perfectly balanced dataset via lightweight augmentations."""
        print("\n" + "="*70)
        print("CREATING BALANCED DATASET")
        print("="*70)
        
        X_class0 = self.X_train[self.y_train == 0]
        X_class1 = self.X_train[self.y_train == 1]
        
        print(f"Original - Class 0: {len(X_class0)}, Class 1: {len(X_class1)}")
        
        def augment_to_target(X_orig, n_target):
            if len(X_orig) >= n_target:
                idx = np.random.choice(len(X_orig), n_target, replace=False)
                return X_orig[idx]
            
            X_result = [X_orig]
            while len(np.vstack(X_result)) < n_target:
                # number we still need (cap to avoid oversampling too big chunks)
                n_needed = n_target - len(np.vstack(X_result))
                idx = np.random.choice(len(X_orig), min(len(X_orig), n_needed))
                
                aug_type = np.random.rand()
                if aug_type < 0.25:
                    # Additive Gaussian noise
                    X_aug = X_orig[idx] + np.random.normal(0, 0.01, (len(idx), X_orig.shape[1]))
                elif aug_type < 0.5:
                    # Multiplicative scaling
                    scale = 1.0 + np.random.uniform(-0.03, 0.03, (len(idx), 1))
                    X_aug = X_orig[idx] * scale
                elif aug_type < 0.75:
                    # Circular shift (time shift)
                    shifts = np.random.randint(-20, 20, len(idx))
                    X_aug = np.array([np.roll(X_orig[i], s) for i, s in zip(idx, shifts)])
                else:
                    # Mild combo: small scale + small noise
                    X_aug = X_orig[idx] * (1.0 + np.random.uniform(-0.02, 0.02, (len(idx), 1)))
                    X_aug += np.random.normal(0, 0.008, X_aug.shape)
                
                X_result.append(X_aug)
            
            X_final = np.vstack(X_result)
            return X_final[:n_target]
        
        X0_bal = augment_to_target(X_class0, self.samples_per_class)
        X1_bal = augment_to_target(X_class1, self.samples_per_class)
        
        print(f"Balanced - Class 0: {len(X0_bal)}, Class 1: {len(X1_bal)}")
        
        X_balanced = np.vstack([X0_bal, X1_bal])
        y_balanced = np.concatenate([np.zeros(self.samples_per_class), np.ones(self.samples_per_class)])
        
        # Shuffle
        idx = np.arange(len(X_balanced))
        np.random.shuffle(idx)
        
        return X_balanced[idx], y_balanced[idx]


    def load_data(self):
        """Load CSV, split, balance train set, and standardize features."""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        df = pd.read_csv(self.dataset_path)
        print(f"Dataset: {df.shape[0]} samples")
        
        flux_cols = [f'flux_{i:04d}' for i in range(self.n_bins)]
        flux_err_cols = [f'flux_err_{i:04d}' for i in range(self.n_bins)]
        X = df[flux_cols].values
        X_err = df[flux_err_cols].values
        y = df['label'].values
        
        metadata_cols = ['toi_name', 'tic', 'label', 'disp', 'period_d', 't0_bjd', 'dur_hr', 'sector']
        metadata = df[metadata_cols]
        
        print("\nOriginal distribution:")
        print(f"  Class 0: {(y==0).sum()}, Class 1: {(y==1).sum()}")
        if (y==0).sum() > 0:
            print(f"  Ratio: {(y==1).sum() / (y==0).sum():.2f}:1")
        
        # Train/test split (keep errors aligned; stratify to preserve class ratio)
        self.X_train, self.X_test, self.y_train, self.y_test, self.X_err_train, self.X_err_test, _, idx_test = train_test_split(
            X, y, X_err, np.arange(len(y)),
            test_size=0.2,
            random_state=42,
            stratify=y
        )
        
        print(f"\nInitial split - Train: {len(self.X_train)}, Test: {len(self.X_test)}")
        
        # Balance training set
        self.X_train, self.y_train = self.create_balanced_dataset()
        
        # Standardize (fit on train, apply to test)
        print("\n" + "="*70)
        print("STANDARDIZATION")
        print("="*70)
        
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print(f"Train: mean={self.X_train.mean():.6f}, std={self.X_train.std():.6f}")
        print(f"Test:  mean={self.X_test.mean():.6f}, std={self.X_test.std():.6f}")
        
        # Reshape for Conv1D: (samples, timesteps, channels)
        self.X_train = self.X_train.reshape(-1, self.n_bins, 1)
        self.X_test = self.X_test.reshape(-1, self.n_bins, 1)
        
        self.metadata_test = metadata.iloc[idx_test].reset_index(drop=True)
        
        print(f"\nFinal - X_train: {self.X_train.shape}, X_test: {self.X_test.shape}")
        print(f"Train dist: 0={( self.y_train==0).sum()}, 1={(self.y_train==1).sum()}")

        self.X_test_orig = self.X_test.copy()
        # Return standardized test for model input, but also return the standardized
        # copy (X_test_orig) so we can inverse-transform for plotting with error bars.


    def build_simple_cnn(self):
        """Simpler CNN to prevent overfitting on small datasets."""
        print("\n" + "="*70)
        print("BUILDING SIMPLIFIED CNN")
        print("="*70)
        
        self.model = models.Sequential([
            layers.Input(shape=(self.n_bins, 1)),
            
            # Feature extraction
            layers.Conv1D(64, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(128, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2),
            layers.Dropout(0.3),
            
            layers.Conv1D(256, kernel_size=3, padding='same', activation='relu'),
            layers.BatchNormalization(),
            layers.GlobalAveragePooling1D(),
            layers.Dropout(0.4),
            
            # Classification head
            layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            layers.Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.001)),
            layers.Dropout(0.2),
            
            layers.Dense(1, activation='sigmoid')
        ])
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0005),
            loss=self.focal_loss,
            metrics=['accuracy',
                    keras.metrics.Precision(name='precision'),
                    keras.metrics.Recall(name='recall'),
                    keras.metrics.AUC(name='auc')]
        )
        
        self.model.summary()
        print("\nUsing Focal Loss (gamma=2.5, alpha=0.75)")


    def train_model(self):
        """Train the model with AUC-centric callbacks."""
        print("\n" + "="*70)
        print("TRAINING")
        print("="*70)
        
        callbacks = [
            EarlyStopping(
                monitor='val_auc',
                patience=20,
                restore_best_weights=True,
                mode='max',
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_auc',
                factor=0.5,
                patience=8,
                min_lr=1e-7,
                mode='max',
                verbose=1
            ),
            ModelCheckpoint(self.output_dir + '/best_model_final.keras',
                monitor='val_auc',
                save_best_only=True,
                mode='max',
                verbose=1
            )
        ]
        
        self.history = self.model.fit(
            self.X_train, self.y_train,
            validation_data=(self.X_test, self.y_test),
            epochs=self.epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )


    def evaluate_with_optimal_threshold(self):
        """Find an optimal threshold from ROC (Youden's J) and evaluate."""
        print("\n" + "="*70)
        print("THRESHOLD OPTIMIZATION & EVALUATION")
        print("="*70)
        
        self.y_pred_proba = self.model.predict(self.X_test, verbose=0).flatten()
        fpr, tpr, thresholds = roc_curve(self.y_test, self.y_pred_proba)
        
        # Youden's J statistic
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        self.optimal_threshold = thresholds[optimal_idx]
        
        print(f"\nOptimal threshold: {self.optimal_threshold:.4f} (default=0.5)")
        print(f"  At this threshold: TPR={tpr[optimal_idx]:.4f}, FPR={fpr[optimal_idx]:.4f}")
        
        # Predictions with optimal vs default thresholds
        self.y_pred_optimal = (self.y_pred_proba >= self.optimal_threshold).astype(int)
        y_pred_default = (self.y_pred_proba >= 0.5).astype(int)
        
        # Metrics
        acc_optimal = accuracy_score(self.y_test, self.y_pred_optimal)
        acc_default = accuracy_score(self.y_test, y_pred_default)
        auc = roc_auc_score(self.y_test, self.y_pred_proba)
        
        print("\nResults:")
        print(f"  AUC-ROC: {auc:.4f}")
        print(f"  Accuracy (default threshold=0.5): {acc_default:.4f} ({acc_default*100:.2f}%)")
        print(f"  Accuracy (optimal threshold={self.optimal_threshold:.4f}): {acc_optimal:.4f} ({acc_optimal*100:.2f}%)")
        
        print("\nWith optimal threshold:")
        print(classification_report(self.y_test, self.y_pred_optimal,
                                    target_names=['Non-Planet', 'Planet'],
                                    digits=4,
                                    zero_division=0))
        
        print("\nPrediction distribution (optimal threshold):")
        print(f"  Predicted 0: {(self.y_pred_optimal == 0).sum()}")
        print(f"  Predicted 1: {(self.y_pred_optimal == 1).sum()}")
        print("True distribution:")
        print(f"  True 0: {(self.y_test == 0).sum()}")
        print(f"  True 1: {(self.y_test == 1).sum()}")


    def plot_lightcurves_with_predictions(self):
        """Plot light curves with error bars and prediction info; save to file."""
        print("\n" + "="*70)
        print(f"PLOTTING LIGHTCURVES WITH PREDICTIONS (n={self.n_samples})")
        print("="*70)
        
        n_samples = min(self.n_samples, len(self.X_test_orig))
        
        # Select diverse samples: correct/incorrect for both classes
        correct_planet = np.where((self.y_test == 1) & (self.y_pred_optimal == 1))[0]
        incorrect_planet = np.where((self.y_test == 1) & (self.y_pred_optimal == 0))[0]
        correct_nonplanet = np.where((self.y_test == 0) & (self.y_pred_optimal == 0))[0]
        incorrect_nonplanet = np.where((self.y_test == 0) & (self.y_pred_optimal == 1))[0]
        
        selected_idx = []
        per_category = max(1, n_samples // 4)
        
        for idx_list in [correct_planet, incorrect_planet, correct_nonplanet, incorrect_nonplanet]:
            if len(idx_list) > 0:
                n_select = min(per_category, len(idx_list))
                selected_idx.extend(np.random.choice(idx_list, n_select, replace=False))
        
        while len(selected_idx) < n_samples:
            remaining = list(set(range(len(self.y_test))) - set(selected_idx))
            if remaining:
                selected_idx.append(np.random.choice(remaining))
            else:
                break
        
        selected_idx = np.array(selected_idx[:n_samples])
        
        # Figure layout
        n_cols = 2
        n_rows = (n_samples + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
        if n_samples == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for plot_i, idx in enumerate(selected_idx):
            ax = axes[plot_i]
            
            # Inverse transform to original scale for plotting
            flux_norm = self.X_test_orig[idx].flatten()
            flux_err = self.X_err_test[idx]
            flux_original = self.scaler.inverse_transform(flux_norm.reshape(1, -1)).flatten()
            
            time_bins = np.arange(len(flux_original))
            
            # Metadata
            toi_name = self.metadata_test.loc[idx, 'toi_name']
            tic = self.metadata_test.loc[idx, 'tic']
            disp = self.metadata_test.loc[idx, 'disp']
            sector = self.metadata_test.loc[idx, 'sector']
            
            true_label = self.y_test[idx]
            pred_label = self.y_pred_optimal[idx]
            pred_prob = self.y_pred_proba[idx]
            
            is_correct = (true_label == pred_label)
            true_str = 'Transit' if true_label == 1 else 'Non-Transit'
            pred_str = 'Transit' if pred_label == 1 else 'Non-Transit'
            
            # Errorbar plot
            ax.errorbar(time_bins, flux_original, yerr=flux_err, fmt='o', markersize=2,
                        ecolor='gray', elinewidth=0.5, capsize=0, alpha=0.6, label='Data')
            
            # Baseline median
            baseline = np.median(flux_original)
            ax.axhline(baseline, linestyle='--', linewidth=1, alpha=0.7, label='Baseline')
            
            ax.set_xlabel('Time Bin', fontsize=10, fontweight='bold')
            ax.set_ylabel('Flux (original scale)', fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=8)
            
            status_symbol = '✓' if is_correct else '✗'
            color = 'green' if is_correct else 'red'
            title = (f'TOI {toi_name} (TIC {tic}, {disp}) - TESS Sector {sector}\n'
                    f'True: {true_str} | Pred: {pred_str} (p={pred_prob:.3f}) {status_symbol}')
            ax.set_title(title, fontsize=10, fontweight='bold', color=color, pad=10)
            
            for spine in ax.spines.values():
                spine.set_edgecolor(color)
                spine.set_linewidth(2.0)
        
        # Hide unused axes
        for j in range(n_samples, len(axes)):
            axes[j].axis('off')
        
        plt.suptitle(f'Sample Light-curve Predictions (Threshold={self.optimal_threshold:.3f})',
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        plt.savefig(self.output_dir + '/' + self.output_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {self.output_dir}/{self.output_path}")
        plt.close()


    def plot_all(self, X_test_orig=None, X_err_test=None, scaler=None):
        """Create and save confusion matrix and training curves. Optionally plot light curves."""
        print("\n" + "="*70)
        print("VISUALIZATIONS")
        print("="*70)
        
        # Confusion matrix (Matplotlib-only)
        cm = confusion_matrix(self.y_test, self.y_pred_optimal)
        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(2),
            yticks=np.arange(2),
            xticklabels=['Non-Planet', 'Planet'],
            yticklabels=['Non-Planet', 'Planet'],
            xlabel='Predicted', ylabel='True',
            title=f'Confusion Matrix (threshold={self.optimal_threshold:.3f})')
        
        # Add counts and percentages
        total = cm.sum()
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                count = cm[i, j]
                pct = (count / total * 100) if total > 0 else 0.0
                ax.text(j, i, f"{count}\n({pct:.1f}%)", ha='center', va='center', color='black', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir + '/confusion_matrix_final.png', dpi=300)
        print(f"Saved: {self.output_dir}/confusion_matrix_final.png")
        plt.close()
        
        # Training history
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [('loss', 'Loss'), ('accuracy', 'Accuracy'),
                ('auc', 'AUC'), ('recall', 'Recall')]
        
        for idx, (metric, title) in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            if metric in self.history.history and f'val_{metric}' in self.history.history:
                ax.plot(self.history.history[metric], label='Train', linewidth=2)
                ax.plot(self.history.history[f'val_{metric}'], label='Val', linewidth=2)
                ax.set_xlabel('Epoch')
                ax.set_ylabel(title)
                ax.set_title(f'{title} vs Epoch', fontweight='bold')
                ax.legend()
                ax.grid(alpha=0.3)
        
        plt.suptitle('Training History - Final Model', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.output_dir + '/training_history_final.png', dpi=300)
        print(f"Saved: {self.output_dir}/training_history_final.png")
        plt.close()
        
        # Optional: light-curve panel
        if X_test_orig is not None and X_err_test is not None and scaler is not None:
            self.plot_lightcurves_with_predictions()


    def main(self):
        self.load_data()
        self.build_simple_cnn()
        self.train_model()
        self.evaluate_with_optimal_threshold()
        self.plot_all(X_test_orig=self.X_test_orig, X_err_test=self.X_err_test, scaler=self.scaler)
        self.model.save(self.output_dir + '/tess_model_final.keras')
        np.save(self.output_dir + '/optimal_threshold.npy', self.optimal_threshold)

        print("\n" + "="*70)
        print("TRAINING COMPLETE!")
        print("="*70)
        print("\nKey improvements:")
        print("  ✓ Perfectly balanced training data")
        print("  ✓ Focal loss for hard examples")
        print("  ✓ Optimal threshold selection")
        print("  ✓ AUC-focused optimization")
        print("\nFiles:")
        print("  - tess_model_final.keras")
        print("  - best_model_final.keras")
        print("  - optimal_threshold.npy")
        print("  - confusion_matrix_final.png")
        print("  - training_history_final.png")
        print("  - sample_lightcurves_predictions.png")
        print("="*70)