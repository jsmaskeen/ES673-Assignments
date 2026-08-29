import argparse
from src.pipeline import ModifiedPerceptronPipeline


def main():
    parser = argparse.ArgumentParser(
        description="Bio-Inspired Modified Perceptron Experiment Pipeline"
    )
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs"
    )
    parser.add_argument(
        "--time-steps",
        type=int,
        default=20,
        help="Number of timesteps for BPTT integration",
    )
    parser.add_argument(
        "--weight-scale",
        type=float,
        default=10.0,
        help="Multiplier for initial raw weights",
    )
    parser.add_argument(
        "--output-neurons",
        type=int,
        default=10,
        help="Number of neurons in the final layer",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="experiment_results",
        help="Directory for reports and outputs",
    )

    args = parser.parse_args()

    pipeline = ModifiedPerceptronPipeline(
        time_steps=args.time_steps,
        epochs=args.epochs,
        weight_scale=args.weight_scale,
        output_neurons=args.output_neurons,
        output_dir=args.output_dir,
    )

    pipeline.run_mnist_experiment()


if __name__ == "__main__":
    main()
