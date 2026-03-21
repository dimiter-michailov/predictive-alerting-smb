from pipelines.forecasting_pipeline import run_forecasting_single_machine
from pipelines.classification_pipeline import run_classifier_single_machine

def ask_user_choice(prompt, valid_choices):
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"Invalid choice. Choose one of: {valid_choices}")

def ask_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt).strip())

            if min_value is not None and value < min_value:
                print(f"Please enter an integer >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter an integer <= {max_value}.")
                continue

            return value

        except ValueError:
            print("Please enter an integer.")

def ask_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt).strip())

            if min_value is not None and value < min_value:
                print(f"Please enter a number >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"Please enter a number <= {max_value}.")
                continue

            return value

        except ValueError:
            print("Please enter a number.")

def main():
    print("Choose pipeline:")
    print("1 - Forecasting model (predict next point, alert above error threshold)")
    print("2 - Binary classification model (anomaly yes/no within next H steps)")

    pipeline_type = ask_user_choice("Enter 1 or 2: ", {"1", "2"})
    machine_id = input("Enter machine id (e.g. machine-1-1): ").strip()

    if pipeline_type == "1":
        print("\nChoose forecasting model:")
        print("1 - hist_gb")
        print("2 - ridge")

        model_choice = ask_user_choice("Enter 1 or 2: ", {"1", "2"})
        model_name = "hist_gb" if model_choice == "1" else "ridge"

        window_size = ask_int("Enter window size: ", min_value=1)
        threshold_quantile = ask_float(
            "Enter threshold quantile in [0, 1] (e.g. 0.99): ",
            min_value=0.0,
            max_value=1.0
        )

        run_forecasting_single_machine(
            machine_id=machine_id,
            model_name=model_name,
            window_size=window_size,
            threshold_quantile=threshold_quantile,
        )

    else:
        window_size = ask_int("Enter window size: ", min_value=1)
        horizon = ask_int("Enter horizon H: ", min_value=1)
        threshold = ask_float(
            "Enter classification threshold (e.g. 0.5): ",
            min_value=0.0,
            max_value=1.0
        )

        run_classifier_single_machine(
            machine_id=machine_id,
            window_size=window_size,
            horizon=horizon,
            threshold=threshold,
        )

if __name__ == "__main__":
    main()