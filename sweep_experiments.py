import os
import csv
import matplotlib.pyplot as plt
import argparse
from src.pipeline import ModifiedPerceptronPipeline


def run_sweep():
    parser = argparse.ArgumentParser(description="Sweep Bio-inspired Constants")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs per run"
    )
    parser.add_argument(
        "--time-steps", type=int, default=20, help="Number of timesteps"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results/sweeps",
        help="Directory for sweep results",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    defaults = {
        "lambda_constant": 150.0,
        "tau_m": 20.0,
        "U": 0.2,
        "gamma": 0.02,
        "surrogate_k": 25.0,
    }

    sweep_ranges = {
        "lambda_constant": [80.0, 150.0, 250.0, 350.0],
        "tau_m": [3.0, 10.0, 20.0, 40.0, 80.0],
        "U": [0.05, 0.15, 0.35, 0.65, 0.95],
        "gamma": [0.001, 0.05, 0.1, 0.15, 0.3, 0.6],
        "surrogate_k": [5.0, 15.0, 25.0, 35.0, 50.0],
    }

    results = []

    csv_path = os.path.join(args.output_dir, "sweep_results.csv")
    with open(csv_path, "w", newline="") as csvfile:
        fieldnames = [
            "swept_param",
            "param_value",
            "lambda_constant",
            "tau_m",
            "U",
            "gamma",
            "surrogate_k",
            "test_accuracy",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for param_to_sweep, values in sweep_ranges.items():
            print(f"Sweeping {param_to_sweep}")

            param_results = []

            for val in values:
                current_params = defaults.copy()
                current_params[param_to_sweep] = val

                run_dir = os.path.join(args.output_dir, f"{param_to_sweep}_{val}")
                os.makedirs(run_dir, exist_ok=True)

                print(f"\n-> Running {param_to_sweep} = {val}...")

                pipeline = ModifiedPerceptronPipeline(
                    time_steps=args.time_steps,
                    epochs=args.epochs,
                    output_dir=run_dir,
                    plot=False,
                    **current_params,
                )

                acc = pipeline.run_mnist_experiment()

                result_row = {
                    "swept_param": param_to_sweep,
                    "param_value": val,
                    "lambda_constant": current_params["lambda_constant"],
                    "tau_m": current_params["tau_m"],
                    "U": current_params["U"],
                    "gamma": current_params["gamma"],
                    "surrogate_k": current_params["surrogate_k"],
                    "test_accuracy": acc,
                }

                writer.writerow(result_row)
                csvfile.flush()
                param_results.append((val, acc))
                results.append(result_row)

            vals, accs = zip(*param_results)
            plt.figure()
            plt.plot(vals, accs, marker="o")
            plt.title(f"Effect of {param_to_sweep} on Accuracy")
            plt.xlabel(param_to_sweep)
            plt.ylabel("Test Accuracy (%)")
            plt.grid(True)
            plt.savefig(
                os.path.join(args.output_dir, f"sweep_plot_{param_to_sweep}.png")
            )
            plt.close()

    print(f"\nAll sweeps completed! Results saved to {args.output_dir}")


if __name__ == "__main__":
    run_sweep()
