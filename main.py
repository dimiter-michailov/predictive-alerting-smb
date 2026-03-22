from pipelines.forecasting_pipeline import run_forecasting_single_machine
from pipelines.classification_pipeline import (
    run_classifier_single_machine,
    run_classifier_multi_machine,
)

AVAILABLE_MACHINE_IDS = (
    [f"machine-1-{i}" for i in range(1, 9)] +
    [f"machine-2-{i}" for i in range(1, 10)] +
    [f"machine-3-{i}" for i in range(1, 12)]
)

def ask_user_choice(prompt, valid_choices):
    while True:
        choice = input(prompt).strip()
        if choice in valid_choices:
            return choice
        print(f"\nInvalid choice. Choose one of: {valid_choices}")

def ask_int(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = int(input(prompt).strip())

            if min_value is not None and value < min_value:
                print(f"\nPlease enter an integer >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"\nPlease enter an integer <= {max_value}.")
                continue

            return value

        except ValueError:
            print("\nPlease enter an integer.")

def ask_float(prompt, min_value=None, max_value=None):
    while True:
        try:
            value = float(input(prompt).strip())

            if min_value is not None and value < min_value:
                print(f"\nPlease enter a number >= {min_value}.")
                continue

            if max_value is not None and value > max_value:
                print(f"\nPlease enter a number <= {max_value}.")
                continue

            return value

        except ValueError:
            print("\nPlease enter a number.")

def print_available_machines():
    print("\nAvailable machines:")
    for index, machine_id in enumerate(AVAILABLE_MACHINE_IDS, start=1):
        print(f"{index:2d} - {machine_id}")

def parse_machine_number_list(raw_text, max_value):
    parts = [part.strip() for part in raw_text.split(",") if part.strip()]
    if not parts:
        raise ValueError("\nPlease enter at least one machine number.")

    numbers = []
    seen = set()

    for part in parts:
        if not part.isdigit():
            raise ValueError("\nMachine numbers must be integers separated by commas.")

        value = int(part)
        if value < 1 or value > max_value:
            raise ValueError(f"\nMachine numbers must be between 1 and {max_value}.")

        if value not in seen:
            seen.add(value)
            numbers.append(value)

    return numbers

def ask_single_machine_id():
    print_available_machines()
    machine_number = ask_int(
        "\nEnter one machine number: ",
        min_value=1,
        max_value=len(AVAILABLE_MACHINE_IDS)
    )
    return AVAILABLE_MACHINE_IDS[machine_number - 1]

def ask_multi_machine_ids():
    print_available_machines()

    while True:
        raw = input(
            "\nEnter machine numbers separated by commas "
            "(e.g. 1,2,3,5): "
        ).strip()

        try:
            machine_numbers = parse_machine_number_list(raw, len(AVAILABLE_MACHINE_IDS))
            return [AVAILABLE_MACHINE_IDS[number - 1] for number in machine_numbers]
        except ValueError as exc:
            print(exc)

def main():
    print("Choose pipeline:")
    print("1 - Binary classification model (predict whether an incident will occur within the next H steps)")
    print("2 - Forecasting/regression model (forecast future metric value(s); alert when prediction error exceeds a threshold)")

    pipeline_type = ask_user_choice("Enter 1 or 2: ", {"1", "2"})

    if pipeline_type == "1":
        print("\nChoose classification run mode:")
        print("1 - Single-machine")
        print("2 - Multi-machine")

        run_mode = ask_user_choice("Enter 1 or 2: ", {"1", "2"})
        window_size = ask_int("Enter window size: ", min_value=1)
        horizon = ask_int("Enter horizon H: ", min_value=1)

        if run_mode == "1":
            machine_id = ask_single_machine_id()

            print("\nRun summary:")
            print("  Mode: single-machine")
            print(f"  Machine: {machine_id}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: {horizon}")

            run_classifier_single_machine(
                machine_id=machine_id,
                window_size=window_size,
                horizon=horizon,
            )
        else:
            machine_ids = ask_multi_machine_ids()

            print("\nRun summary:")
            print("  Mode: multi-machine")
            print(f"  Machines ({len(machine_ids)}): {', '.join(machine_ids)}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: {horizon}")

            run_classifier_multi_machine(
                machine_ids=machine_ids,
                window_size=window_size,
                horizon=horizon,
            )

    else:
        print("\nChoose forecasting mode:")
        print("1 - HistGB single-point forecast")
        print("2 - HistGB recursive H-step forecast path")

        forecast_mode = ask_user_choice("Enter 1 or 2: ", {"1", "2"})
        window_size = ask_int("Enter window size: ", min_value=1)
        threshold_quantile = ask_float(
            "\nEnter threshold quantile in [0, 1] (e.g. 0.99): ",
            min_value=0.0,
            max_value=1.0
        )
        machine_id = ask_single_machine_id()

        if forecast_mode == "1":
            print("\nRun summary:")
            print("  Mode: forecasting")
            print("  Forecast type: HistGB single-point")
            print(f"  Machine: {machine_id}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: 1")
            print(f"  Threshold quantile: {threshold_quantile}")

            run_forecasting_single_machine(
                machine_id=machine_id,
                model_name="hist_gb_single",
                window_size=window_size,
                horizon=1,
                threshold_quantile=threshold_quantile,
            )

        else:
            horizon = ask_int("Enter recursive forecast horizon H: ", min_value=1)

            print("\nRun summary:")
            print("  Mode: forecasting")
            print("  Forecast type: HistGB recursive H-step")
            print(f"  Machine: {machine_id}")
            print(f"  Window size: {window_size}")
            print(f"  Horizon: {horizon}")
            print(f"  Threshold quantile: {threshold_quantile}")

            run_forecasting_single_machine(
                machine_id=machine_id,
                model_name="hist_gb_recursive_h",
                window_size=window_size,
                horizon=horizon,
                threshold_quantile=threshold_quantile,
            )

if __name__ == "__main__":
    main()