# src/predict.py
"""
Prediction script for wool quality classification
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import config


class WoolPredictor:
    """Make predictions on new wool images"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.class_names = config.CLASSES
        self.load_model()
        
    def load_model(self):
        """Load trained model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        
        Args:
            image_path: path to image file
            
        Returns:
            preprocessed image array
        """
        img = load_img(image_path, target_size=config.IMG_SIZE)
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        
        return img_array, img
    
    def predict_single(self, image_path, show_confidence=True):
        """
        Predict quality of a single image
        
        Args:
            image_path: path to image file
            show_confidence: whether to show confidence scores
            
        Returns:
            prediction: class name
            confidence: confidence score
        """
        # Preprocess image
        img_array, original_img = self.preprocess_image(image_path)
        
        # Make prediction
        predictions = self.model.predict(img_array, verbose=0)
        predicted_class = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class]
        
        class_name = self.class_names[predicted_class]
        
        if show_confidence:
            print(f"\nPrediction: {class_name}")
            print(f"Confidence: {confidence:.2%}")
            print(f"\nClass probabilities:")
            for i, cls in enumerate(self.class_names):
                print(f"  {cls}: {predictions[0][i]:.2%}")
        
        return class_name, confidence, predictions[0]
    
    def predict_batch(self, image_paths):
        """
        Predict quality for multiple images
        
        Args:
            image_paths: list of image paths
            
        Returns:
            results: list of (image_path, prediction, confidence) tuples
        """
        results = []
        
        print(f"\nProcessing {len(image_paths)} images...")
        
        for image_path in image_paths:
            try:
                class_name, confidence, _ = self.predict_single(image_path, show_confidence=False)
                results.append((image_path, class_name, confidence))
                print(f"✓ {os.path.basename(image_path)}: {class_name} ({confidence:.2%})")
            except Exception as e:
                print(f"✗ Error processing {image_path}: {str(e)}")
                results.append((image_path, "Error", 0.0))
        
        return results
    
    def visualize_prediction(self, image_path, save_path=None):
        """
        Visualize prediction with image
        
        Args:
            image_path: path to image
            save_path: path to save visualization
        """
        class_name, confidence, probabilities = self.predict_single(image_path, show_confidence=False)
        _, original_img = self.preprocess_image(image_path)
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Show image
        axes[0].imshow(original_img)
        axes[0].set_title(f'Prediction: {class_name}\nConfidence: {confidence:.2%}', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Show probabilities
        colors = ['red' if cls == 'Bad' else 'green' for cls in self.class_names]
        axes[1].barh(self.class_names, probabilities, color=colors, alpha=0.7)
        axes[1].set_xlabel('Probability', fontsize=12)
        axes[1].set_title('Class Probabilities', fontsize=14, fontweight='bold')
        axes[1].set_xlim([0, 1])
        
        for i, (cls, prob) in enumerate(zip(self.class_names, probabilities)):
            axes[1].text(prob + 0.02, i, f'{prob:.2%}', 
                        va='center', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Visualization saved to: {save_path}")
        
        plt.show()
    
    def predict_directory(self, directory_path, save_results=True):
        """
        Predict all images in a directory
        
        Args:
            directory_path: path to directory containing images
            save_results: whether to save results to CSV
            
        Returns:
            results: list of predictions
        """
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if f.lower().endswith(image_extensions)
        ]
        
        if not image_paths:
            print(f"No images found in {directory_path}")
            return []
        
        # Make predictions
        results = self.predict_batch(image_paths)
        
        # Save results
        if save_results:
            import pandas as pd
            df = pd.DataFrame(results, columns=['Image Path', 'Prediction', 'Confidence'])
            csv_path = os.path.join(directory_path, 'predictions.csv')
            df.to_csv(csv_path, index=False)
            print(f"\nResults saved to: {csv_path}")
        
        return results


def main():
    """Main prediction pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Predict Wool Quality')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model (.keras file)')
    parser.add_argument('--image_path', type=str,
                       help='Path to single image for prediction')
    parser.add_argument('--directory_path', type=str,
                       help='Path to directory containing images')
    parser.add_argument('--visualize', action='store_true',
                       help='Visualize predictions')
    
    args = parser.parse_args()
    
    # Create predictor
    predictor = WoolPredictor(args.model_path)
    
    if args.image_path:
        # Single image prediction
        if args.visualize:
            predictor.visualize_prediction(args.image_path)
        else:
            predictor.predict_single(args.image_path)
            
    elif args.directory_path:
        # Directory prediction
        predictor.predict_directory(args.directory_path)
    else:
        print("Please provide either --image_path or --directory_path")


if __name__ == "__main__":
    main()
