# src/model.py
"""
Deep Learning models for wool quality classification
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.applications import (
    VGG16, ResNet50, EfficientNetB0, MobileNetV2
)
import config


class WoolClassifier:
    """Build and compile different CNN architectures"""
    
    def __init__(self, model_name='efficientnet', input_shape=(224, 224, 3), num_classes=2):
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_cnn_scratch(self):
        """
        Build a CNN from scratch
        Good for understanding but may not perform as well as transfer learning
        """
        model = models.Sequential([
            # Block 1
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 2
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 3
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Block 4
            layers.Conv2D(256, (3, 3), activation='relu'),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Dropout(0.25),
            
            # Dense layers
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_vgg16(self):
        """
        Build VGG16 with transfer learning
        Good accuracy but slower training
        """
        base_model = VGG16(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_resnet50(self):
        """
        Build ResNet50 with transfer learning
        Excellent for complex patterns
        """
        base_model = ResNet50(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_efficientnet(self):
        """
        Build EfficientNetB0 with transfer learning
        RECOMMENDED: Best balance of accuracy and efficiency
        """
        base_model = EfficientNetB0(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.3),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build_mobilenet(self):
        """
        Build MobileNetV2 with transfer learning
        Lightweight and fast, good for deployment
        """
        base_model = MobileNetV2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dense(512, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(256, activation='relu'),
            layers.BatchNormalization(),
            layers.Dropout(0.5),
            layers.Dense(self.num_classes, activation='softmax')
        ])
        
        return model
    
    def build(self):
        """Build the specified model architecture"""
        print(f"Building {self.model_name} model...")
        
        if self.model_name == 'cnn_scratch':
            self.model = self.build_cnn_scratch()
        elif self.model_name == 'vgg16':
            self.model = self.build_vgg16()
        elif self.model_name == 'resnet50':
            self.model = self.build_resnet50()
        elif self.model_name == 'efficientnet':
            self.model = self.build_efficientnet()
        elif self.model_name == 'mobilenet':
            self.model = self.build_mobilenet()
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
        
        return self.model
    
    def compile(self, learning_rate=config.LEARNING_RATE):
        """Compile the model with optimizer, loss, and metrics"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        print("Model compiled successfully!")
        return self.model
    
    def summary(self):
        """Print model summary"""
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        return self.model.summary()
    
    def unfreeze_layers(self, num_layers=None):
        """
        Unfreeze layers for fine-tuning
        
        Args:
            num_layers: number of layers to unfreeze from the end (None = all)
        """
        if self.model is None:
            raise ValueError("Model not built yet. Call build() first.")
        
        if self.model_name == 'cnn_scratch':
            print("Cannot unfreeze layers for CNN from scratch")
            return
        
        # Get the base model (first layer in Sequential)
        base_model = self.model.layers[0]
        
        if num_layers is None:
            base_model.trainable = True
            print(f"All layers unfrozen for fine-tuning")
        else:
            base_model.trainable = True
            # Freeze all layers except the last num_layers
            for layer in base_model.layers[:-num_layers]:
                layer.trainable = False
            print(f"Last {num_layers} layers unfrozen for fine-tuning")
        
        # Recompile with lower learning rate for fine-tuning
        self.compile(learning_rate=config.LEARNING_RATE / 10)


def main():
    """Test model building"""
    print("Testing all model architectures...\n")
    
    for model_name in config.AVAILABLE_MODELS:
        print(f"\n{'='*50}")
        print(f"Testing: {model_name}")
        print('='*50)
        
        classifier = WoolClassifier(model_name=model_name)
        model = classifier.build()
        classifier.compile()
        
        print(f"\nModel: {model_name}")
        print(f"Total parameters: {model.count_params():,}")
        print(f"Trainable parameters: {sum([tf.size(w).numpy() for w in model.trainable_weights]):,}")


if __name__ == "__main__":
    main()
