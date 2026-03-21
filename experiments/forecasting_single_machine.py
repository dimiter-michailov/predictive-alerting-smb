from pipelines.forecasting_pipeline import run_forecasting_single_machine

if __name__ == "__main__":
    run_forecasting_single_machine(
        machine_id="machine-1-1",
        model_name="hist_gb",
        window_size=60,
        threshold_quantile=0.99,
    )