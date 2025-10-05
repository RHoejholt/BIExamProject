from src.models.train import Trainer

def main():
    t = Trainer()
    print("Training binary first-kill model...")
    info = t.train_binary()
    print("Done. Artifact:", info.get("artifact_path"))
    print("Metrics:", info.get("metrics_path"))

if __name__ == "__main__":
    main()
