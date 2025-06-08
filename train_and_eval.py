import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Load Dataset
data = np.load('pneumoniamnist.npz')
x_train, y_train = data['train_images'], data['train_labels']
x_val, y_val = data['val_images'], data['val_labels']
x_test, y_test = data['test_images'], data['test_labels']

# Expand channels and resize
def prepare(images):
    images = np.repeat(images[..., np.newaxis], 3, axis=-1)
    return tf.image.resize(images, [224, 224]) / 255.0

x_train = prepare(x_train)
x_val = prepare(x_val)
x_test = prepare(x_test)

# Compute class weights
class_weights = compute_class_weight(class_weight='balanced',
                                     classes=np.unique(y_train),
                                     y=y_train.flatten())
class_weights = {i: w for i, w in enumerate(class_weights)}

# Build Model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# Compile
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)

# Train
history = model.fit(x_train, y_train, epochs=70, batch_size=64,
                    validation_data=(x_val, y_val),
                    class_weight=class_weights,
                    callbacks=[early_stop])

# Evaluate
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\n✅ Test Accuracy: {test_acc*100:.2f}%")

# Plot Accuracy and Loss
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Accuracy over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("training_plots.png")
plt.show()

# Classification Report & Confusion Matrix
y_pred = (model.predict(x_test) > 0.5).astype(int)
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred))

plt.figure(figsize=(6, 5))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
