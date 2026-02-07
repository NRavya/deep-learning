# quick_start.py
"""
Quick start script for wool quality classification
This script automates the setup and initial run
"""

import os
import sys
import subprocess
import platform


def print_header(text):
    """Print formatted header"""
    print("\n" + "="*70)
    print(text.center(70))
    print("="*70 + "\n")


def check_python_version():
    """Check if Python version is compatible"""
    print_header("Checking Python Version")
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        sys.exit(1)
    
    print("✓ Python version is compatible")


def create_virtual_environment():
    """Create virtual environment"""
    print_header("Creating Virtual Environment")
    
    if os.path.exists('venv'):
        print("Virtual environment already exists")
        return
    
    try:
        subprocess.run([sys.executable, '-m', 'venv', 'venv'], check=True)
        print("✓ Virtual environment created successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error creating virtual environment: {e}")
        sys.exit(1)


def get_pip_command():
    """Get the correct pip command for the platform"""
    if platform.system() == "Windows":
        return os.path.join('venv', 'Scripts', 'pip.exe')
    else:
        return os.path.join('venv', 'bin', 'pip')


def install_dependencies():
    """Install required dependencies"""
    print_header("Installing Dependencies")
    
    pip_cmd = get_pip_command()
    
    # Upgrade pip
    print("Upgrading pip...")
    try:
        subprocess.run([pip_cmd, 'install', '--upgrade', 'pip'], check=True)
        print("✓ pip upgraded")
    except subprocess.CalledProcessError as e:
        print(f"⚠ Warning: Could not upgrade pip: {e}")
    
    # Install requirements
    print("\nInstalling dependencies from requirements.txt...")
    try:
        subprocess.run([pip_cmd, 'install', '-r', 'requirements.txt'], check=True)
        print("✓ Dependencies installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing dependencies: {e}")
        sys.exit(1)


def create_directory_structure():
    """Create necessary directories"""
    print_header("Creating Directory Structure")
    
    directories = [
        'data',
        'data/dataset',
        'models',
        'results',
        'notebooks'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ Created: {directory}")


def check_dataset():
    """Check if dataset exists"""
    print_header("Checking Dataset")
    
    dataset_dir = os.path.join('data', 'dataset')
    categories = ['Lattice', 'Solid', 'Stripped', 'Printed']
    classes = ['Good', 'Bad']
    
    dataset_exists = True
    total_images = 0
    
    for category in categories:
        for class_name in classes:
            path = os.path.join(dataset_dir, category, class_name)
            
            if os.path.exists(path):
                image_count = len([f for f in os.listdir(path) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                total_images += image_count
                print(f"✓ Found {image_count} images in {category}/{class_name}")
            else:
                print(f"❌ Missing: {category}/{class_name}")
                dataset_exists = False
    
    if dataset_exists and total_images > 0:
        print(f"\n✓ Dataset found with {total_images} total images")
        return True
    else:
        print("\n⚠ Dataset not found or incomplete")
        print("\nPlease organize your dataset in the following structure:")
        print("""
data/dataset/
├── Lattice/
│   ├── Good/
│   └── Bad/
├── Solid/
│   ├── Good/
│   └── Bad/
├── Stripped/
│   ├── Good/
│   └── Bad/
└── Printed/
    ├── Good/
    └── Bad/
        """)
        return False


def run_test_training():
    """Run a quick test training"""
    print_header("Running Test Training (5 epochs)")
    
    python_cmd = sys.executable
    
    try:
        subprocess.run([
            python_cmd, 'main.py',
            '--model', 'efficientnet',
            '--epochs', '5'
        ], check=True)
        print("\n✓ Test training completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error during test training: {e}")
        return False
    
    return True


def print_next_steps():
    """Print next steps for the user"""
    print_header("Setup Complete!")
    
    print("Next steps:")
    print("\n1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    
    print("\n2. Train the full model:")
    print("   python main.py --model efficientnet --epochs 50")
    
    print("\n3. Compare all models:")
    print("   python compare_models.py --epochs 30")
    
    print("\n4. Evaluate a trained model:")
    print("   python src/evaluate.py --model_path models/<model_dir>/best_model.keras --save_results")
    
    print("\n5. Make predictions:")
    print("   python src/predict.py --model_path models/<model_dir>/best_model.keras --image_path <image.jpg> --visualize")
    
    print("\n6. Open Jupyter notebook for experimentation:")
    print("   jupyter notebook notebooks/experimentation.ipynb")
    
    print("\n" + "="*70)
    print("For more information, see README.md and VSCODE_SETUP_GUIDE.md")
    print("="*70 + "\n")


def main():
    """Main setup function"""
    print("""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                                                                  ║
    ║          Wool Quality Classification - Quick Start              ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """)
    
    # Check Python version
    check_python_version()
    
    # Create virtual environment
    create_virtual_environment()
    
    # Install dependencies
    install_dependencies()
    
    # Create directory structure
    create_directory_structure()
    
    # Check dataset
    dataset_ready = check_dataset()
    
    if dataset_ready:
        # Ask user if they want to run test training
        print_header("Ready to Train")
        response = input("Would you like to run a quick test training (5 epochs)? [y/N]: ")
        
        if response.lower() in ['y', 'yes']:
            run_test_training()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()
