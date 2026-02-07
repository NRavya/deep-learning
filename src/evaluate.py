# src/evaluate.py
"""
Model evaluation and performance metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, 
    accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, auc
)
from tensorflow import keras
import config
from src.data_loader import WoolDataLoader


class ModelEvaluator:
    """Evaluate trained models"""
    
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load saved model"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from: {self.model_path}")
        self.model = keras.models.load_model(self.model_path)
        print("Model loaded successfully!")
        
    def evaluate(self, X_test, y_test):
        """
        Evaluate model on test set
        
        Args:
            X_test: test images
            y_test: test labels
            
        Returns:
            metrics: dictionary of evaluation metrics
        """
        print("\nEvaluating model on test set...")
        
        # Get predictions
        y_pred_proba = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='binary')
        recall = recall_score(y_test, y_pred, average='binary')
        f1 = f1_score(y_test, y_pred, average='binary')
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
        
        print("\n" + "="*50)
        print("Test Set Performance")
        print("="*50)
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print("="*50)
        
        return metrics, y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, save_path=None):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Bad', 'Good'],
                   yticklabels=['Bad', 'Good'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to: {save_path}")
        
        plt.show()
        
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        """Plot ROC curve"""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1])
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ROC curve saved to: {save_path}")
        
        plt.show()
        
    def classification_report_text(self, y_true, y_pred):
        """Generate detailed classification report"""
        report = classification_report(
            y_true, y_pred,
            target_names=['Bad', 'Good'],
            digits=4
        )
        
        print("\n" + "="*50)
        print("Classification Report")
        print("="*50)
        print(report)
        
        return report
    
    def visualize_predictions(self, X_test, y_test, y_pred, num_samples=16, save_path=None):
        """Visualize sample predictions"""
        # Select random samples
        indices = np.random.choice(len(X_test), size=min(num_samples, len(X_test)), replace=False)
        
        rows = 4
        cols = 4
        fig, axes = plt.subplots(rows, cols, figsize=(15, 15))
        
        for idx, ax in enumerate(axes.flat):
            if idx >= len(indices):
                ax.axis('off')
                continue
                
            i = indices[idx]
            
            img = X_test[i]

            if hasattr(img,"detach"):
                img = img.detach().cpu().numpy()
            
            if img.shape[0] == 3:
                img = np.transpose(img,(1,2,0))
            
            mean = np.array([0.485,0.456,0.406])
            std = np.array([0.229,0.224,.0225])
            img = (img * std) + mean

            img = np.clip(img,0,1)
            print("shape:", img.shape,
                  "min:", img.min(),
                  "max:", img.max(),
                    "mean:", img.mean())

            ax.imshow(img)
            
            true_label = 'Good' if y_test[i] == 1 else 'Bad'
            pred_label = 'Good' if y_pred[i] == 1 else 'Bad'
            
            color = 'green' if y_test[i] == y_pred[i] else 'red'
            ax.set_title(f'True: {true_label}\nPred: {pred_label}', 
                        color=color, fontsize=12 , fontweight = 'bold')
            ax.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction visualization saved to: {save_path}")
        
        plt.show()


def main():
    """Main evaluation pipeline"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate Wool Quality Classification Model')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to saved model (.keras file)')
    parser.add_argument('--save_results', action='store_true',
                       help='Save evaluation results and plots')
    
    args = parser.parse_args()
    
    # Load test data
    print("Loading dataset...")
    loader = WoolDataLoader()
    X, y, df = loader.load_dataset()
    _, _, X_test, _, _, y_test = loader.split_data(X, y)
    
    # Create evaluator
    evaluator = ModelEvaluator(args.model_path)
    
    # Evaluate model
    metrics, y_pred, y_pred_proba = evaluator.evaluate(X_test, y_test)
    
    # Generate classification report
    report = evaluator.classification_report_text(y_test, y_pred)
    
    # Determine save directory
    if args.save_results:
        model_dir = os.path.dirname(args.model_path)
        results_dir = os.path.join(model_dir, 'evaluation_results')
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = None
    
    # Plot confusion matrix
    cm_path = os.path.join(results_dir, 'confusion_matrix.png') if results_dir else None
    evaluator.plot_confusion_matrix(y_test, y_pred, save_path=cm_path)
    
    # Plot ROC curve
    roc_path = os.path.join(results_dir, 'roc_curve.png') if results_dir else None
    evaluator.plot_roc_curve(y_test, y_pred_proba, save_path=roc_path)
    
    # Visualize predictions
    pred_path = os.path.join(results_dir, 'sample_predictions.png') if results_dir else None
    evaluator.visualize_predictions(X_test, y_test, y_pred, save_path=pred_path)
    
    # Save metrics to file
    if results_dir:
        metrics_path = os.path.join(results_dir, 'metrics.txt')
        with open(metrics_path, 'w') as f:
            f.write("Model Evaluation Metrics\n")
            f.write("="*50 + "\n\n")
            for key, value in metrics.items():
                f.write(f"{key}: {value:.4f}\n")
            f.write("\n" + "="*50 + "\n\n")
            f.write("Classification Report\n")
            f.write("="*50 + "\n")
            f.write(report)
        
        print(f"\nEvaluation results saved to: {results_dir}")


if __name__ == "__main__":
    main()
