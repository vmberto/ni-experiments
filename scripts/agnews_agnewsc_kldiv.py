"""
AG News KL Divergence Comparison Script

Compares in-distribution AG News with corrupted AG News-C variants
using text autoencoder latent representations and KL divergence.
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import KFold
from sklearn.feature_extraction.text import TfidfVectorizer
import tensorflow as tf
from keras import callbacks

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.consts import AGNEWS_CORRUPTIONS
from dataset.ood_characterization import KLDivergenceComparer
from models.text_autoencoder import TextAutoencoder


class AGNewsDatasetLoader:
    """Load and preprocess AG News dataset."""
    
    def __init__(self, vocab_size=10000, max_len=200):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.vectorizer = TfidfVectorizer(max_features=vocab_size)
        
    def load_agnews(self):
        """
        Load AG News dataset.
        Returns: texts_train, labels_train, texts_test, labels_test
        """
        try:
            # Try loading from tensorflow_datasets
            import tensorflow_datasets as tfds
            
            train_ds = tfds.load('ag_news_subset', split='train', as_supervised=True)
            test_ds = tfds.load('ag_news_subset', split='test', as_supervised=True)
            
            texts_train = []
            labels_train = []
            for text, label in tfds.as_numpy(train_ds):
                texts_train.append(text.decode('utf-8'))
                labels_train.append(int(label))
            
            texts_test = []
            labels_test = []
            for text, label in tfds.as_numpy(test_ds):
                texts_test.append(text.decode('utf-8'))
                labels_test.append(int(label))
            
            return texts_train, labels_train, texts_test, labels_test
            
        except Exception as e:
            print(f"Error loading AG News: {e}")
            print("Please ensure tensorflow_datasets is installed and AG News is available.")
            raise
    
    def load_corrupted_agnews(self, corruption_type):
        """
        Load corrupted AG News variant.
        
        Args:
            corruption_type: e.g., 'typo_1', 'synonym_3', etc.
        
        Returns:
            texts_corrupted, labels_corrupted
        """
        # Check if corruption files exist
        corruption_dir = project_root / 'dataset' / 'agnewsdataset-c'
        corruption_file = corruption_dir / f'ag_news_{corruption_type}.csv'
        
        if not corruption_file.exists():
            raise FileNotFoundError(
                f"Corruption file not found: {corruption_file}\n"
                f"Please generate AG News-C corruptions first."
            )
        
        # Load from CSV
        df = pd.read_csv(corruption_file)
        texts_corrupted = df['text'].tolist()
        labels_corrupted = df['label'].tolist()
        
        return texts_corrupted, labels_corrupted
    
    def texts_to_sequences(self, texts, fit=False):
        """
        Convert texts to TF-IDF vectors for autoencoder input.
        
        Args:
            texts: List of text strings
            fit: If True, fit the vectorizer on these texts
        
        Returns:
            Dense numpy array of TF-IDF features
        """
        if fit:
            tfidf_matrix = self.vectorizer.fit_transform(texts)
        else:
            tfidf_matrix = self.vectorizer.transform(texts)
        
        # Convert sparse to dense
        return tfidf_matrix.toarray().astype('float32')


def build_text_autoencoder(input_dim, latent_dim=64):
    """
    Build a simple autoencoder for TF-IDF vectors.
    
    Args:
        input_dim: TF-IDF vocabulary size
        latent_dim: Dimension of latent space
    
    Returns:
        encoder, decoder, autoencoder models
    """
    from keras import layers, models
    
    # Encoder
    encoder_input = layers.Input(shape=(input_dim,), name='tfidf_input')
    x = layers.Dense(512, activation='relu')(encoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent_space')(x)
    
    encoder = models.Model(encoder_input, latent, name='encoder')
    
    # Decoder
    decoder_input = layers.Input(shape=(latent_dim,), name='latent_input')
    x = layers.Dense(256, activation='relu')(decoder_input)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    decoder_output = layers.Dense(input_dim, activation='sigmoid')(x)
    
    decoder = models.Model(decoder_input, decoder_output, name='decoder')
    
    # Autoencoder
    autoencoder_input = layers.Input(shape=(input_dim,))
    encoded = encoder(autoencoder_input)
    decoded = decoder(encoded)
    autoencoder = models.Model(autoencoder_input, decoded, name='autoencoder')
    
    return encoder, decoder, autoencoder


def prepare_tf_dataset(X, batch_size=128, shuffle=False):
    """Convert numpy array to tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, X))  # Input = output for autoencoder
    
    if shuffle:
        dataset = dataset.shuffle(buffer_size=1000)
    
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset


def main():
    # Configuration
    KFOLD_N_SPLITS = 10
    VOCAB_SIZE = 10000
    MAX_LEN = 200
    LATENT_DIM = 64
    BATCH_SIZE = 128
    EPOCHS = 50
    
    # Setup output directories
    output_dir = project_root / 'output' / 'agnews'
    models_dir = output_dir / 'models'
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = output_dir / f'agnews_autoencoder_kldiv_{timestamp}.csv'
    
    print("=" * 70)
    print("AG News → AG News-C KL Divergence Comparison")
    print("=" * 70)
    print(f"Timestamp: {timestamp}")
    print(f"Results file: {results_file}")
    print(f"K-Folds: {KFOLD_N_SPLITS}")
    print(f"Vocab size: {VOCAB_SIZE}")
    print(f"Latent dim: {LATENT_DIM}")
    print("=" * 70)
    
    # Load dataset
    print("\n📚 Loading AG News dataset...")
    loader = AGNewsDatasetLoader(vocab_size=VOCAB_SIZE, max_len=MAX_LEN)
    
    try:
        texts_train, labels_train, texts_test, labels_test = loader.load_agnews()
        print(f"✓ Loaded {len(texts_train)} training samples")
        print(f"✓ Loaded {len(texts_test)} test samples")
    except Exception as e:
        print(f"❌ Failed to load AG News: {e}")
        return
    
    # Convert texts to TF-IDF features
    print("\n🔤 Converting texts to TF-IDF features...")
    X_train = loader.texts_to_sequences(texts_train, fit=True)
    X_test = loader.texts_to_sequences(texts_test, fit=False)
    input_dim = X_train.shape[1]
    
    print(f"✓ Training shape: {X_train.shape}")
    print(f"✓ Test shape: {X_test.shape}")
    print(f"✓ Input dimension: {input_dim}")
    
    # Prepare K-Fold splits
    kf = KFold(n_splits=KFOLD_N_SPLITS, shuffle=True, random_state=42)
    splits = list(enumerate(kf.split(X_train)))
    
    # Results storage
    results = []
    
    # Train autoencoder for each fold
    for fold, (train_index, val_index) in splits:
        print(f"\n{'='*70}")
        print(f"FOLD {fold + 1}/{KFOLD_N_SPLITS}")
        print(f"{'='*70}")
        
        # Split data
        X_train_fold = X_train[train_index]
        X_val_fold = X_train[val_index]
        
        # Prepare datasets
        train_ds = prepare_tf_dataset(X_train_fold, BATCH_SIZE, shuffle=True)
        val_ds = prepare_tf_dataset(X_val_fold, BATCH_SIZE, shuffle=False)
        test_ds = prepare_tf_dataset(X_test, BATCH_SIZE, shuffle=False)
        
        # Build model
        print("\n🏗️  Building autoencoder...")
        encoder, decoder, autoencoder = build_text_autoencoder(input_dim, LATENT_DIM)
        
        # Compile
        autoencoder.compile(
            optimizer='adam',
            loss='mse',
            metrics=['mae']
        )
        
        print(f"✓ Autoencoder built")
        print(f"  - Encoder params: {encoder.count_params():,}")
        print(f"  - Decoder params: {decoder.count_params():,}")
        print(f"  - Total params: {autoencoder.count_params():,}")
        
        # Train
        print(f"\n🚀 Training autoencoder (fold {fold + 1})...")
        history = autoencoder.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS,
            callbacks=[
                callbacks.EarlyStopping(
                    patience=5,
                    monitor='val_loss',
                    restore_best_weights=True,
                    verbose=1
                ),
                callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=3,
                    min_lr=1e-6,
                    verbose=1
                )
            ],
            verbose=1
        )
        
        epochs_trained = len(history.history['loss'])
        final_train_loss = history.history['loss'][-1]
        final_val_loss = history.history['val_loss'][-1]
        
        print(f"\n✓ Training completed")
        print(f"  - Epochs: {epochs_trained}")
        print(f"  - Final train loss: {final_train_loss:.6f}")
        print(f"  - Final val loss: {final_val_loss:.6f}")
        
        # Save models
        fold_model_dir = models_dir / f'fold_{fold}'
        fold_model_dir.mkdir(exist_ok=True)
        
        autoencoder.save(fold_model_dir / 'autoencoder.keras')
        encoder.save(fold_model_dir / 'encoder.keras')
        decoder.save(fold_model_dir / 'decoder.keras')
        pd.DataFrame(history.history).to_csv(
            fold_model_dir / 'training_history.csv',
            index=False
        )
        
        print(f"✓ Models saved to: {fold_model_dir}")
        
        # Evaluate on corruptions
        print(f"\n📊 Evaluating on {len(AGNEWS_CORRUPTIONS)} corruptions...")
        
        # Get clean latent representations
        latent_clean = encoder.predict(X_test, verbose=0)
        
        for i, corruption_type in enumerate(AGNEWS_CORRUPTIONS, 1):
            print(f"[{i}/{len(AGNEWS_CORRUPTIONS)}] {corruption_type}...", end=' ')
            
            try:
                # Load corrupted texts
                texts_corrupted, labels_corrupted = loader.load_corrupted_agnews(corruption_type)
                
                # Convert to TF-IDF
                X_corrupted = loader.texts_to_sequences(texts_corrupted, fit=False)
                
                # Get latent representations
                latent_corrupted = encoder.predict(X_corrupted, verbose=0)
                
                # Calculate KL divergence
                comparer = KLDivergenceComparer(epsilon=1e-10, method='flatten')
                kld = comparer.compare(latent_clean, latent_corrupted)
                
                # Store result
                result = {
                    "fold": fold + 1,
                    "corruption_type": corruption_type,
                    "kl_divergence": kld,
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                    "epochs_trained": epochs_trained,
                    "timestamp": timestamp,
                    "n_clean_samples": len(X_test),
                    "n_corrupted_samples": len(X_corrupted)
                }
                
                results.append(result)
                print(f"KL = {kld:.6f} ✓")
                
                # Save incrementally
                pd.DataFrame(results).to_csv(results_file, index=False)
                
            except FileNotFoundError as e:
                print(f"SKIP (file not found)")
                continue
            except Exception as e:
                print(f"ERROR: {e}")
                continue
        
        print(f"\n✓ Fold {fold + 1} completed")
    
    # Final summary
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETED")
    print("=" * 70)
    
    if len(results) > 0:
        results_df = pd.DataFrame(results)
        
        # Calculate summary statistics
        summary = results_df.groupby('corruption_type')['kl_divergence'].agg([
            'mean', 'std', 'min', 'max', 'count'
        ]).round(6)
        
        summary_file = output_dir / f'agnews_autoencoder_kldiv_{timestamp}_summary.csv'
        summary.to_csv(summary_file)
        
        print(f"\n📈 Top 10 Most OOD Corruptions (by mean KL divergence):")
        print("-" * 70)
        top = summary.sort_values('mean', ascending=False).head(10)
        for idx, (corruption, row) in enumerate(top.iterrows(), 1):
            print(f"  {idx:2d}. {corruption:30s} {row['mean']:8.6f} ± {row['std']:7.6f}")
        
        print(f"\n📈 Top 10 Least OOD Corruptions (by mean KL divergence):")
        print("-" * 70)
        bottom = summary.sort_values('mean', ascending=True).head(10)
        for idx, (corruption, row) in enumerate(bottom.iterrows(), 1):
            print(f"  {idx:2d}. {corruption:30s} {row['mean']:8.6f} ± {row['std']:7.6f}")
        
        print(f"\n💾 Files saved:")
        print(f"  - Results: {results_file}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Models: {models_dir}")
        
        print(f"\n✅ Total results: {len(results)}")
        print(f"✅ Unique corruptions evaluated: {len(summary)}")
    else:
        print("\n⚠️  No results generated. Check if corruption files exist.")
        print(f"Expected location: {project_root / 'dataset' / 'agnewsdataset-c'}")


if __name__ == "__main__":
    main()

