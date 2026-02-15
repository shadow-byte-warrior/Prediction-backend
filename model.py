
from tensorflow.keras import layers, Model, regularizers

def build_seizure_model(input_shape):
    """
    Build CNN-LSTM model with Attention for seizure detection.
    
    Architecture:
    - Multi-scale CNN for pattern extraction
    - Bidirectional LSTM for temporal learning
    - Attention mechanism to focus on important parts
    - Dense layers for classification
    """
    
    inputs = layers.Input(shape=input_shape, name='eeg_input')
    
    # ===== CNN BLOCK: Extract Local Patterns =====
    # Small kernel - catches quick spikes
    conv1 = layers.Conv1D(32, kernel_size=3, padding='same', activation='relu')(inputs)
    conv1 = layers.BatchNormalization()(conv1)
    conv1 = layers.MaxPooling1D(pool_size=2)(conv1)
    conv1 = layers.Dropout(0.2)(conv1)
    
    # Medium kernel - catches medium patterns
    conv2 = layers.Conv1D(64, kernel_size=5, padding='same', activation='relu')(conv1)
    conv2 = layers.BatchNormalization()(conv2)
    conv2 = layers.MaxPooling1D(pool_size=2)(conv2)
    conv2 = layers.Dropout(0.2)(conv2)
    
    # Larger kernel - catches slower waves
    conv3 = layers.Conv1D(128, kernel_size=7, padding='same', activation='relu')(conv2)
    conv3 = layers.BatchNormalization()(conv3)
    conv3 = layers.MaxPooling1D(pool_size=2)(conv3)
    conv3 = layers.Dropout(0.3)(conv3)
    
    # ===== LSTM BLOCK: Learn Time Patterns =====
    lstm1 = layers.Bidirectional(
        layers.LSTM(64, return_sequences=True, dropout=0.2)
    )(conv3)
    
    lstm2 = layers.Bidirectional(
        layers.LSTM(32, return_sequences=True, dropout=0.2)
    )(lstm1)
    
    # ===== ATTENTION BLOCK: Focus on Important Parts =====
    attention = layers.MultiHeadAttention(
        num_heads=4, key_dim=32, dropout=0.1
    )(lstm2, lstm2)
    attention = layers.LayerNormalization()(attention + lstm2)  # Skip connection
    
    # ===== POOLING: Combine All Information =====
    avg_pool = layers.GlobalAveragePooling1D()(attention)
    max_pool = layers.GlobalMaxPooling1D()(attention)
    concat = layers.Concatenate()([avg_pool, max_pool])
    
    # ===== CLASSIFICATION HEAD =====
    dense1 = layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(concat)
    dense1 = layers.BatchNormalization()(dense1)
    dense1 = layers.Dropout(0.5)(dense1)
    
    dense2 = layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(0.01))(dense1)
    dense2 = layers.Dropout(0.3)(dense2)
    
    # Output: probability of seizure
    outputs = layers.Dense(1, activation='sigmoid', name='seizure_output')(dense2)
    
    model = Model(inputs=inputs, outputs=outputs, name='SeizureDetector')
    
    return model
