import os
import sys
import subprocess

from predict import predict_image
from model_utils import load_checkpoint
from config import DEVICE, OUTPUT_DIR, MODELS


def menu():
    print("\n--- Canadian Rockies Wildlife Compare CNN ---")
    print("Available Actions:")
    print("[1] Train")
    print("[2] Validate")
    print("[3] Compare Results")
    print("[4] View Grad-CAM Overlay")
    print("[5] Exit")
    return input("Select an option: ").strip()

def model_selection():
    print("\nAvailable models:")
    for i, model_name in enumerate(MODELS, start=1):
        print(f"[{i}] {model_name}")

    choice = input("Select a model number: ").strip()

    try:
        model_name = MODELS[int(choice) - 1]
        return model_name
    except (ValueError, IndexError):
        print("Invalid model selection.")
        return None

def input_image():
    image_path = input("Enter image path: ").strip().strip('"')

    if not os.path.exists(image_path):
        print("Image file not found.")
        return None

    return image_path

def get_checkpoint_path(model_name):
    return os.path.join(OUTPUT_DIR, model_name, "model.pth")

def run_script(script_name, args):
    command = [sys.executable, script_name] + args

    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nError running {script_name}: {e}")
    except FileNotFoundError:
        print(f"\nCould not find script: {script_name}")

def train():
    model_name = model_selection()
    if not model_name:
        return

    run_script("train.py", ["--model", model_name])

def validate():
    model_name = model_selection()
    if not model_name:
        return

    checkpoint_path = get_checkpoint_path(model_name)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the model first.")
        return

    print(f"\nRunning validation for model: {model_name}")
    run_script("evaluate.py", ["--model", model_name])

def compare_results():
    image_path = input_image()
    if not image_path:
        return

    print(f"\nComparing predictions for: {image_path}")

    any_success = False

    for model_name in MODELS:
        checkpoint_path = get_checkpoint_path(model_name)
        print(f"\n--- {model_name} ---")

        if not os.path.exists(checkpoint_path):
            print("Checkpoint not found. Train this model first.")
            continue

        try:
            model, class_names, _ = load_checkpoint(checkpoint_path, DEVICE)
            predictions = predict_image(model, image_path, class_names, DEVICE)

            if not predictions:
                print("No predictions returned.")
                continue

            any_success = True

            for label, confidence in predictions:
                print(f"{label}: {confidence:.4f}")

        except Exception as e:
            print(f"Error comparing with {model_name}: {e}")

    if not any_success:
        print("\nNo model results could be generated.")

def gradcam():
    model_name = model_selection()
    if not model_name:
        return

    checkpoint_path = get_checkpoint_path(model_name)
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found: {checkpoint_path}")
        print("Train the model first.")
        return

    image_path = input_image()
    if not image_path:
        return

    print(f"\nGenerating Grad-CAM for model: {model_name}")
    run_script(
        "gradcam.py",
        ["--checkpoint", checkpoint_path, "--image", image_path]
    )

def main():
    while True:
        choice = menu()

        if choice == "1":
            train()
        elif choice == "2":
            validate()
        elif choice == "3":
            compare_results()
        elif choice == "4":
            gradcam()
        elif choice == "5":
            print("Exiting.")
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    main()