import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from algorithms.backprop import MLPClassifier, evaluate_backprop, train_backprop
from algorithms.diffusion import (
    DenoisingDiffusionClassifier,
    evaluate_diffusion,
    sample_diffusion,
    train_diffusion,
)
from algorithms.forward_forward import (
    ForwardForward,
    UnsupervisedForwardForward,
    evaluate_forward_forward,
    evaluate_unsupervised_forward_forward,
    goodness,
    train_forward_forward,
    train_unsupervised_forward_forward,
)
from algorithms.noprop import NoProp, evaluate_noprop, train_noprop
from utils.data import make_loaders, make_synthetic_2d_loader
from utils.metrics import get_device, seed_everything, timed_run
from utils.plotting import (
    plot_2d_decision_boundaries,
    plot_comparative_benchmark,
    plot_energy_and_memory_comparison,
    plot_ff_positive_negative_comparison,
    plot_goodness_comparative_histograms,
    plot_goodness_histograms,
    plot_goodness_relationships,
    plot_hybrid_mask_generation,
    plot_layer_representations,
    plot_noprop_vs_diffusion_denoising,
    plot_unsupervised_goodness_histograms,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Learning Paradigms Benchmarking: Backprop, FF (Supervised & Unsupervised), NoProp, and Diffusion."
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument("--train-subset", type=int, default=2048)
    parser.add_argument("--test-subset", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--mlp-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer dimensions for Backprop MLP.",
    )
    parser.add_argument(
        "--ff-hidden-dims",
        type=int,
        nargs="+",
        default=[512, 256, 128],
        help="Hidden layer dimensions for Forward-Forward networks.",
    )
    parser.add_argument(
        "--ff-threshold",
        type=float,
        default=2.0, # (Hinton 2022 2.0)
        help="Layer-level threshold theta for Forward-Forward.",
    )
    parser.add_argument(
        "--ff-goodness-modes",
        type=str,
        nargs="+",
        default=[
            "squared",
            "negative_squared",
            "sparse",
            "margin",
            "spectral",
            "hebbian_wire",
        ],
    )
    parser.add_argument(
        "--ff-hard-negatives",  #  Hinton:  Instead of pairing an
                                #  image with a randomly chosen incorrect label during 
                                #  the negative pass, it actively searches for and 
                                #  trains against the single most
                                #  confusable wrong class for each image
        action="store_true",
        help="Enable hard negative mining for Supervised FF.",
    )
    parser.add_argument(
        "--ff-include-first-layer",
        action="store_true",
        default=True,
        help="Include first layer when accumulating goodness for Supervised FF evaluation.",
    )
    parser.add_argument(
        "--noprop-blocks",
        type=int,
        default=4,
        help="Number of independent blocks in NoProp-DT.",
    )
    parser.add_argument(
        "--noprop-hidden-dim",
        type=int,
        default=140,
        help="Hidden dimension per block in NoProp-DT.",
    )
    parser.add_argument(
        "--diffusion-hidden-dim",
        type=int,
        default=288,
        help="Hidden dimension for Denoising Diffusion Classifier.",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="iso-parameter",
        choices=["iso-parameter", "optimised", "optimized"],
        help="Configuration preset: 'iso-parameter' (~567k matched params across all models) or 'optimised'/'optimized' (small efficient ~230k params while preserving accuracy within +-1%%).",
    )
    parser.add_argument(
        "--include-unsupervised-ff",
        action="store_true",
        default=True,
        help="Include Hinton's unsupervised hybrid FF.",
    )
    parser.add_argument(
        "--run-2d-demo",
        action="store_true",
        default=True,
        help="Train & plot 2D synthetic decision landscapes.",
    )
    parser.add_argument(
        "--device", type=str, default="auto", choices=["auto", "cpu", "cuda"]
    )
    parser.add_argument("--plot-dir", type=str, default="plots")
    parser.add_argument("--output", type=str, default="results.json")
    parser.add_argument(
        "--sample-output", type=str, default="plots/diffusion_samples.png"
    )
    parser.add_argument("--sample-class", type=int, default=4)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()

    if args.preset in ("optimised", "optimized"):
        passed_args = set(sys.argv[1:])
        if "--mlp-hidden-dims" not in passed_args:
            args.mlp_hidden_dims = [256, 128]
        if "--ff-hidden-dims" not in passed_args:
            args.ff_hidden_dims = [256, 128]
        if "--noprop-blocks" not in passed_args:
            args.noprop_blocks = 3
        if "--noprop-hidden-dim" not in passed_args:
            args.noprop_hidden_dim = 80
        if "--diffusion-hidden-dim" not in passed_args:
            args.diffusion_hidden_dim = 160

    return args


def extract_layer_representation_data(
    backprop, unsupervised_spectral, noprop, train_loader, test_loader, device
):
    """Extracts layer-by-layer representation statistics across paradigms."""
    layer_data = {}

    # Backprop layer-wise linear probe separability
    print("[Representations] Probing Backprop layer hierarchy...")
    backprop.eval()

    def extract_bp_feats(loader):
        feats_by_layer = [[] for _ in range(len(backprop.hidden_dims))]
        y_list = []
        with torch.no_grad():
            for imgs, lbls in loader:
                imgs = imgs.to(device)
                layer_feats = backprop.get_layer_features(imgs)
                for idx, lf in enumerate(layer_feats):
                    feats_by_layer[idx].append(lf)
                y_list.append(lbls.to(device))
        cat_feats = [torch.cat(f_list) for f_list in feats_by_layer]
        cat_y = torch.cat(y_list)
        return cat_feats, cat_y

    tr_feats, tr_y = extract_bp_feats(train_loader)
    te_feats, te_y = extract_bp_feats(test_loader)

    def quick_linear_probe(tr_x, tr_y, te_x, te_y):
        clf = torch.nn.Linear(tr_x.size(1), 10).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=0.01)
        for _ in range(60):
            opt.zero_grad()
            pred = clf(tr_x)
            loss = F.cross_entropy(pred, tr_y)
            loss.backward()
            opt.step()
        with torch.no_grad():
            acc = (clf(te_x).argmax(1) == te_y).float().mean().item() * 100.0
        return acc

    bp_layer_accs = []
    for idx in range(len(backprop.hidden_dims)):
        dim = backprop.hidden_dims[idx]
        acc_l = quick_linear_probe(tr_feats[idx], tr_y, te_feats[idx], te_y)
        bp_layer_accs.append((f"Layer {idx+1} ({dim})", round(acc_l, 1)))
    full_acc = evaluate_backprop(backprop, test_loader, device) * 100.0
    bp_layer_accs.append(("Final Output", round(full_acc, 1)))
    layer_data["backprop_layer_accs"] = bp_layer_accs

    # Unsupervised FF (Spectral) layer goodness margins
    if unsupervised_spectral is not None:
        print(
            "[Representations] Probing Unsupervised FF (Spectral) multi-scale margins..."
        )
        unsupervised_spectral.eval()
        with torch.no_grad():
            imgs, _ = next(iter(test_loader))
            imgs = imgs.to(device)
            pos_x, neg_x = unsupervised_spectral.make_positive_negative(imgs)
            l_margins = []
            pos_cur, neg_cur = pos_x, neg_x
            for idx, l in enumerate(unsupervised_spectral.layers):
                pos_act, pos_cur = l(pos_cur)
                neg_act, neg_cur = l(neg_cur)
                g_p = goodness(pos_act, "spectral").mean().item()
                g_n = goodness(neg_act, "spectral").mean().item()
                l_margins.append((f"Layer {idx+1}", round(g_p, 3), round(g_n, 3)))
            layer_data["ff_spectral_margins"] = l_margins

    # NoProp target space reverse diffusion error
    if noprop is not None:
        print("[Representations] Probing NoProp target simplex trajectory...")
        noprop.eval()
        with torch.no_grad():
            errors = [0.0] * (noprop.num_blocks + 1)
            count = 0
            for imgs, lbls in test_loader:
                imgs = imgs.to(device)
                lbls = lbls.to(device)
                b = imgs.size(0)
                y_target = F.one_hot(lbls, 10).float()
                state = torch.randn(b, 10, device=device)
                errors[0] += F.mse_loss(state, y_target).item() * b
                for b_idx in range(noprop.num_blocks):
                    state = noprop.step(imgs, state, b_idx)
                    errors[b_idx + 1] += F.mse_loss(state, y_target).item() * b
                count += b
            step_errors = [("z0 (Prior Noise)", round(errors[0] / count, 3))]
            for b_idx in range(noprop.num_blocks):
                step_errors.append(
                    (
                        f"z{b_idx+1} (Block {b_idx+1})",
                        round(errors[b_idx + 1] / count, 3),
                    )
                )
            layer_data["noprop_step_errors"] = step_errors

    return layer_data


def run_2d_experiment(args, device, plot_dir: Path):
    print("\nRunning 2D Synthetic Toy Dataset Experiment")
    n_classes = 3
    tr_loader, te_loader, tr_ds, te_ds = make_synthetic_2d_loader(
        n_samples=1200, n_classes=n_classes, noise=0.15, batch_size=64
    )

    # Backprop on 2D (4-layer proportional: 2 -> 64 -> 32 -> 16 -> 3)
    bp_2d = MLPClassifier(input_size=2, hidden_dims=[64, 32, 16], num_classes=n_classes)
    train_backprop(bp_2d, tr_loader, epochs=10, lr=1e-2, device=device)
    bp_acc = evaluate_backprop(bp_2d, te_loader, device=device)
    print(f"[2D] Backprop test accuracy: {bp_acc:.3f}")

    # Supervised Forward-Forward on 2D (input_dim is 2 coords + n_classes one-hot)
    ff_2d = ForwardForward(
        input_dim=2 + n_classes,
        hidden_dims=[64, 32, 16],
        num_classes=n_classes,
        mode="squared",
    )
    train_forward_forward(
        ff_2d, tr_loader, epochs=10, lr=1e-2, threshold=args.ff_threshold, device=device
    )
    ff_acc = evaluate_forward_forward(
        ff_2d, te_loader, threshold=args.ff_threshold, device=device
    )
    print(f"[2D] Supervised Forward-Forward (squared) test accuracy: {ff_acc:.3f}")

    # NoProp on 2D
    np_2d = NoProp(
        blocks=args.noprop_blocks, input_dim=2, hidden_dim=64, num_classes=n_classes
    )
    train_noprop(np_2d, tr_loader, epochs=10, lr=1e-2, device=device)
    np_acc = evaluate_noprop(np_2d, te_loader, device=device)
    print(f"[2D] NoProp test accuracy: {np_acc:.3f}")

    # Plot 2D Decision Boundaries
    plot_2d_decision_boundaries(
        bp_2d,
        ff_2d,
        np_2d,
        tr_ds,
        plot_dir / "decision_boundaries_2d.png",
        device=device,
    )


def run(args):
    seed_everything(args.seed)
    device = get_device(args.device == "cuda" if args.device != "auto" else True)
    print(f"Using device: {device}")

    train_loader, test_loader = make_loaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_subset=args.train_subset,
        test_subset=args.test_subset,
    )

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # Generate Hinton Mask & Positive/Negative Comparison Figures
    print("\nGenerating Data Plots")
    plot_hybrid_mask_generation(train_loader, plot_dir / "hybrid_mask_generation.png")
    plot_ff_positive_negative_comparison(
        train_loader, plot_dir / "ff_positive_negative.png"
    )

    def cleanup_memory():
        import gc

        gc.collect()
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()

    # Warmup GPU to ensure active compute P-state (P0) and initialized CUDA kernels
    if device.type == "cuda":
        _dummy = torch.randn(512, 512, device=device)
        for _ in range(10):
            _dummy = torch.matmul(_dummy, _dummy)
        cleanup_memory()

    results = {}

    # Normal MLP (Backprop)
    print("\n [1/5] Training Normal MLP (Backpropagation) ")
    cleanup_memory()
    backprop = MLPClassifier(
        input_size=784, hidden_dims=args.mlp_hidden_dims, num_classes=10
    )
    elapsed, m_info, _ = timed_run(
        "Backprop",
        lambda: train_backprop(backprop, train_loader, args.epochs, args.lr, device),
        backprop,
        device=device,
    )
    results["backprop_time_s"] = elapsed
    results["backprop_energy_joules"] = m_info["energy_joules"]
    results["backprop_avg_power_w"] = m_info["avg_power_w"]
    results["backprop_peak_memory_mb"] = m_info["peak_memory_mb"]
    results["backprop_accuracy"] = evaluate_backprop(backprop, test_loader, device)
    backprop.cpu()
    cleanup_memory()

    # Supervised Forward-Forward (and goodness mode exploration)
    print("\n [2/5] Training Supervised Forward-Forward ")
    supervised_ff_models = {}
    for mode in args.ff_goodness_modes:
        print(f"\nSupervised FF (Goodness mode: {mode})")
        cleanup_memory()
        ff = ForwardForward(
            input_dim=784, hidden_dims=args.ff_hidden_dims, num_classes=10, mode=mode
        )
        elapsed, m_info, _ = timed_run(
            f"Supervised FF ({mode})",
            lambda cur_ff=ff: train_forward_forward(
                cur_ff,
                train_loader,
                args.epochs,
                args.lr,
                args.ff_threshold,
                device,
                hard_negatives=args.ff_hard_negatives,
            ),
            ff,
            device=device,
        )
        acc = evaluate_forward_forward(
            ff,
            test_loader,
            args.ff_threshold,
            device,
            skip_first=not args.ff_include_first_layer,
        )
        results[f"supervised_ff_{mode}_time_s"] = elapsed
        results[f"supervised_ff_{mode}_energy_joules"] = m_info["energy_joules"]
        results[f"supervised_ff_{mode}_avg_power_w"] = m_info["avg_power_w"]
        results[f"supervised_ff_{mode}_peak_memory_mb"] = m_info["peak_memory_mb"]
        results[f"supervised_ff_{mode}_accuracy"] = acc
        plot_goodness_histograms(
            ff,
            train_loader,
            mode,
            plot_dir / f"supervised_ff_{mode}_goodness.png",
            threshold=args.ff_threshold,
            device=device,
        )
        ff.cpu()
        supervised_ff_models[mode] = ff
        cleanup_memory()

    # Generate comparative goodness histograms and inter-metric relationships
    if supervised_ff_models:
        print("\nGenerating Comparative Goodness Plots")
        plot_goodness_comparative_histograms(
            supervised_ff_models,
            test_loader,
            plot_dir / "goodness_functions_comparison_histograms.png",
            threshold=args.ff_threshold,
            device=device,
        )
        ref_model = supervised_ff_models.get(
            "squared", list(supervised_ff_models.values())[0]
        )
        plot_goodness_relationships(
            ref_model,
            test_loader,
            args.ff_goodness_modes,
            plot_dir / "goodness_functions_relationships.png",
            device=device,
        )
        for m in supervised_ff_models.values():
            m.cpu()
        cleanup_memory()

    # Unsupervised Forward-Forward (Hinton hybrid masks across goodness modes)
    unsupervised_ff_models = {}
    if args.include_unsupervised_ff:
        print(
            "\n=== [3/5] Training Unsupervised Forward-Forward (Hinton Hybrid Masks across Goodness Modes) ==="
        )
        for mode in args.ff_goodness_modes:
            print(f"\nUnsupervised FF (Goodness mode: {mode})")
            cleanup_memory()
            unsupervised_ff = UnsupervisedForwardForward(
                input_dim=784,
                hidden_dims=args.ff_hidden_dims,
                num_classes=10,
                mode=mode,
            )
            elapsed, m_info, _ = timed_run(
                f"Unsupervised FF ({mode})",
                lambda cur_unsup=unsupervised_ff: train_unsupervised_forward_forward(
                    cur_unsup,
                    train_loader,
                    args.epochs,
                    args.lr,
                    args.ff_threshold,
                    device,
                ),
                unsupervised_ff,
                device=device,
            )
            acc = evaluate_unsupervised_forward_forward(
                unsupervised_ff, train_loader, test_loader, device
            )
            results[f"unsupervised_ff_{mode}_time_s"] = elapsed
            results[f"unsupervised_ff_{mode}_energy_joules"] = m_info["energy_joules"]
            results[f"unsupervised_ff_{mode}_avg_power_w"] = m_info["avg_power_w"]
            results[f"unsupervised_ff_{mode}_peak_memory_mb"] = m_info["peak_memory_mb"]
            results[f"unsupervised_ff_{mode}_accuracy"] = acc
            print(
                f"[Unsupervised FF ({mode})] Final linear probe test accuracy: {acc:.4f}"
            )
            plot_unsupervised_goodness_histograms(
                unsupervised_ff,
                train_loader,
                mode,
                plot_dir / f"unsupervised_ff_{mode}_goodness.png",
                threshold=args.ff_threshold,
                device=device,
            )
            unsupervised_ff.cpu()
            unsupervised_ff_models[mode] = unsupervised_ff
            cleanup_memory()

    # NoProp-DT
    print("\n [4/5] Training NoProp-DT (Discrete-Time Diffusion Blocks) ")
    cleanup_memory()
    noprop = NoProp(
        blocks=args.noprop_blocks,
        input_dim=784,
        hidden_dim=args.noprop_hidden_dim,
        num_classes=10,
    )
    elapsed, m_info, _ = timed_run(
        "NoProp",
        lambda: train_noprop(noprop, train_loader, args.epochs, args.lr, device),
        noprop,
        device=device,
    )
    results["noprop_time_s"] = elapsed
    results["noprop_energy_joules"] = m_info["energy_joules"]
    results["noprop_avg_power_w"] = m_info["avg_power_w"]
    results["noprop_peak_memory_mb"] = m_info["peak_memory_mb"]
    results["noprop_accuracy"] = evaluate_noprop(noprop, test_loader, device)
    plot_noprop_vs_diffusion_denoising(
        noprop,
        test_loader,
        plot_dir / "noprop_vs_diffusion_denoising.png",
        device=device,
    )
    cleanup_memory()

    # Denoising Diffusion Baseline
    print("\n [5/5] Training Denoising Diffusion Classifier Baseline ")
    cleanup_memory()
    diffusion = DenoisingDiffusionClassifier(
        input_dim=784, hidden_dim=args.diffusion_hidden_dim, num_classes=10
    )
    elapsed, m_info, _ = timed_run(
        "Diffusion",
        lambda: train_diffusion(
            diffusion, train_loader, args.epochs, args.lr, device, total_steps=30
        ),
        diffusion,
        device=device,
    )
    results["diffusion_time_s"] = elapsed
    results["diffusion_energy_joules"] = m_info["energy_joules"]
    results["diffusion_avg_power_w"] = m_info["avg_power_w"]
    results["diffusion_peak_memory_mb"] = m_info["peak_memory_mb"]
    results["diffusion_accuracy"] = evaluate_diffusion(
        diffusion, test_loader, device, total_steps=30
    )

    if args.sample_output:
        sample_diffusion(
            diffusion, args.sample_class, 16, device, args.sample_output, total_steps=30
        )
    diffusion.cpu()
    cleanup_memory()

    # Extract layer-by-layer representations
    backprop.to(device)
    spectral_model = unsupervised_ff_models.get("spectral")
    if spectral_model is None and unsupervised_ff_models:
        spectral_model = list(unsupervised_ff_models.values())[0]
    if spectral_model is not None:
        spectral_model.to(device)
    noprop.to(device)
    layer_data = extract_layer_representation_data(
        backprop=backprop,
        unsupervised_spectral=spectral_model,
        noprop=noprop,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
    )
    plot_layer_representations(layer_data, plot_dir / "layer_representations.png")

    # Energy & Memory comparison
    plot_energy_and_memory_comparison(
        results, plot_dir / "energy_and_memory_comparison.png"
    )

    plot_comparative_benchmark(results, plot_dir / "comparative_benchmark.png")

    if args.run_2d_demo:
        run_2d_experiment(args, device, plot_dir)

    print("\n" + "=" * 50)
    print("Results:")
    print(json.dumps(results, indent=2))

    Path(args.output).write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run(parse_args())
