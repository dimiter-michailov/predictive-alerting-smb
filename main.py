from pipelines.forecasting_pipeline import run_forecasting_single_machine
from pipelines.classification_pipeline import (
    run_classifier_single_machine,
    run_classifier_multi_machine,
)

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

def ask_non_empty_string(prompt):
    while True:
        value = input(prompt).strip()
        if value:
            return value
        print("Input cannot be empty.")

def parse_machine_ids(raw_text):
    machine_ids = [item.strip() for item in raw_text.split(",") if item.strip()]
    unique_machine_ids = []

    for machine_id in machine_ids:
        if machine_id not in unique_machine_ids:
            unique_machine_ids.append(machine_id)

    return unique_machine_ids

def ask_machine_id():
    return ask_non_empty_string("Enter machine id (e.g. machine-1-1): ")

def ask_machine_ids():
    print("\nChoose multi-machine input mode:")
    print("1 - Enter comma-separated machine ids")
    print("2 - Enter machine ids one by one")

    mode = ask_user_choice("Enter 1 or 2: ", {"1", "2"})

    if mode == "1":
        while True:
            raw = input(
                "Enter machine ids separated by commas\n"
                "(e.g. machine-1-1, machine-1-2, machine-1-3): "
            ).strip()

            machine_ids = parse_machine_ids(raw)
            if machine_ids:
                return machine_ids

            print("Please enter at least one machine id.")

    machine_ids = []
    print("\nEnter machine ids one by one. Press Enter on an empty line when done.")

    while True:
        prompt = f"Machine {len(machine_ids) + 1}: "
        value = input(prompt).strip()

        if not value:
            if machine_ids:
                return machine_ids
            print("Please enter at least one machine id.")
            continue

        if value in machine_ids:
            print("This machine id is already added.")
            continue

        machine_ids.append(value)

def main():
    print("Choose pipeline:")
    print("1 - Binary classification model (anomaly within next H steps)")
    print("2 - Forecasting model (predict next point, alert above error threshold)")

    pipeline_type = ask_user_choice("Enter 1 or 2: ", {"1", "2"})

    if pipeline_type == "2":
        machine_id = ask_machine_id()

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
        print("\nChoose classification run mode:")
        print("1 - Single-machine")
        print("2 - Multi-machine")

        run_mode = ask_user_choice("Enter 1 or 2: ", {"1", "2"})

        window_size = ask_int("Enter window size: ", min_value=1)
        horizon = ask_int("Enter horizon H: ", min_value=1)

        if run_mode == "1":
            machine_id = ask_machine_id()

            print("\nRun summary:")
            print(f"  Mode: single-machine")
            print(f"  Machine: {machine_id}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: {horizon}")

            run_classifier_single_machine(
                machine_id=machine_id,
                window_size=window_size,
                horizon=horizon,
            )

        else:
            machine_ids = ask_machine_ids()

            print("\nRun summary:")
            print(f"  Mode: multi-machine")
            print(f"  Machines ({len(machine_ids)}): {', '.join(machine_ids)}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: {horizon}")

            run_classifier_multi_machine(
                machine_ids=machine_ids,
                window_size=window_size,
                horizon=horizon,
            )

if __name__ == "__main__":
    main()