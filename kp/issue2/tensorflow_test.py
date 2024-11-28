import tensorflow as tf
import numpy as np

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu'),  # Hidden layer
    tf.keras.layers.Dense(1, activation='sigmoid') # Output layer
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create a sample dataset
# 100 samples, each with 5 features
X = np.random.rand(100, 5)
# Binary labels (0 or 1)
y = np.random.randint(0, 2, 100)

# Train the model
print("Training the model...")
model.fit(X, y, epochs=10, batch_size=8)

# Test the model on new data
test_data = np.random.rand(5, 5)  # 5 samples, each with 5 features
predictions = model.predict(test_data)

# Display predictions
print("\nPredictions on test data:")
print(predictions)
