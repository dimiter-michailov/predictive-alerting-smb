from pipelines.classification_pipeline import run_classifier_single_machine

if __name__ == "__main__":
    run_classifier_single_machine(
        machine_id="machine-1-1",
        window_size=60,
        horizon=30,
        threshold=0.5,
    )