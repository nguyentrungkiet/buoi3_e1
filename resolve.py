import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class DigitRecognizer:
    def __init__(self):
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load and prepare MNIST data"""
        print("Đang tải dữ liệu...")
        
        # Load training data
        train_df = pd.read_csv('dataset/train.csv')
        test_df = pd.read_csv('dataset/test.csv')
        
        # Separate features and labels
        X = train_df.drop('label', axis=1).values
        y = train_df['label'].values
        X_test = test_df.values
        
        # Reshape to 28x28 images
        X = X.reshape(-1, 28, 28, 1)
        X_test = X_test.reshape(-1, 28, 28, 1)
        
        # Normalize pixel values to [0, 1]
        X = X.astype('float32') / 255.0
        X_test = X_test.astype('float32') / 255.0
        
        # Convert labels to categorical
        y = keras.utils.to_categorical(y, 10)
        
        # Split training data into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Kích thước dữ liệu:")
        print(f"X_train: {X_train.shape}")
        print(f"X_val: {X_val.shape}")
        print(f"X_test: {X_test.shape}")
        
        return X_train, X_val, y_train, y_val, X_test
    
    def create_data_augmentation(self):
        """Create data augmentation layer"""
        data_augmentation = keras.Sequential([
            layers.RandomRotation(0.1),
            layers.RandomZoom(0.1),
            layers.RandomTranslation(0.1, 0.1),
        ])
        return data_augmentation
    
    def build_model(self):
        """Build CNN model with Conv-ReLU-Pool architecture"""
        print("Đang xây dựng mô hình CNN...")
        
        # Data augmentation
        data_augmentation = self.create_data_augmentation()
        
        model = keras.Sequential([
            # Data augmentation layer
            data_augmentation,
            
            # First Convolutional Block
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.BatchNormalization(),
            layers.Conv2D(32, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Second Convolutional Block
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Third Convolutional Block
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.25),
            
            # Flatten and Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(10, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def train_model(self, X_train, X_val, y_train, y_val, epochs=50):
        """Train the CNN model"""
        print("Đang huấn luyện mô hình...")
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_accuracy',
                patience=10,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.2,
                patience=5,
                min_lr=0.0001
            )
        ]
        
        # Train model
        history = self.model.fit(
            X_train, y_train,
            batch_size=128,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        self.history = history
        return history
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            print("Chưa có lịch sử huấn luyện!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def evaluate_model(self, X_val, y_val):
        """Evaluate model performance"""
        print("\nĐánh giá mô hình:")
        
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true_classes = np.argmax(y_val, axis=1)
        
        # Calculate accuracy
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        print(f"Validation Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_true_classes, y_pred_classes))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true_classes, y_pred_classes)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=range(10), yticklabels=range(10))
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return accuracy
    
    def predict_and_save(self, X_test):
        """Make predictions and save to submission file"""
        print("Đang dự đoán và lưu kết quả...")
        
        # Make predictions
        predictions = self.model.predict(X_test)
        predicted_classes = np.argmax(predictions, axis=1)
        
        # Create submission file
        submission = pd.DataFrame({
            'ImageId': range(1, len(predicted_classes) + 1),
            'Label': predicted_classes
        })
        
        submission.to_csv('submission.csv', index=False)
        print("Đã lưu kết quả vào submission.csv")
        
        return submission
    
    def visualize_predictions(self, X_test, num_samples=10):
        """Visualize some predictions"""
        predictions = self.model.predict(X_test[:num_samples])
        predicted_classes = np.argmax(predictions, axis=1)
        
        fig, axes = plt.subplots(2, 5, figsize=(12, 6))
        axes = axes.flatten()
        
        for i in range(num_samples):
            axes[i].imshow(X_test[i].reshape(28, 28), cmap='gray')
            axes[i].set_title(f'Predicted: {predicted_classes[i]}')
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig('predictions_sample.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main execution function"""
    print("="*50)
    print("NHẬN DẠNG CHỮ SỐ MNIST VỚI CNN")
    print("="*50)
    
    # Initialize recognizer
    recognizer = DigitRecognizer()
    
    # Load data
    X_train, X_val, y_train, y_val, X_test = recognizer.load_data()
    
    # Build model
    model = recognizer.build_model()
    print("\nKiến trúc mô hình:")
    model.summary()
    
    # Train model
    history = recognizer.train_model(X_train, X_val, y_train, y_val, epochs=50)
    
    # Plot training history
    recognizer.plot_training_history()
    
    # Evaluate model
    accuracy = recognizer.evaluate_model(X_val, y_val)
    
    # Make predictions and save submission
    submission = recognizer.predict_and_save(X_test)
    
    # Visualize some predictions
    recognizer.visualize_predictions(X_test)
    
    print(f"\nKết quả cuối cùng:")
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Đã lưu submission.csv với {len(submission)} predictions")

if __name__ == "__main__":
    main()
