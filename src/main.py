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
    parser.add_argument(
        "--lambda-constant", type=float, default=150.0, help="Length constant for distance attenuation"
    )
    parser.add_argument(
        "--tau-m", type=float, default=20.0, help="Membrane time constant"
    )
    parser.add_argument(
        "--U", type=float, default=0.2, help="Neurotransmitter utilization factor"
    )
    parser.add_argument(
        "--gamma", type=float, default=0.02, help="Threshold jump"
    )
    parser.add_argument(
        "--surrogate-k", type=float, default=25.0, help="Surrogate gradient slope"
    )

    args = parser.parse_args()

    pipeline = ModifiedPerceptronPipeline(
        time_steps=args.time_steps,
        epochs=args.epochs,
        weight_scale=args.weight_scale,
        output_neurons=args.output_neurons,
        output_dir=args.output_dir,
        lambda_constant=args.lambda_constant,
        tau_m=args.tau_m,
        U=args.U,
        gamma=args.gamma,
        surrogate_k=args.surrogate_k,
    )

    pipeline.run_mnist_experiment()


if __name__ == "__main__":
    main()
