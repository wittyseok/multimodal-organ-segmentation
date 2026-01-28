#!/usr/bin/env python3
"""
Multi-Modal Medical Image Segmentation Pipeline

Unified entry point for training, evaluation, and inference.

Usage:
    # Training
    python main.py --mode train --config configs/default.yaml

    # Evaluation
    python main.py --mode eval --config configs/default.yaml --checkpoint outputs/best_model.pth

    # Inference
    python main.py --mode inference --config configs/default.yaml --checkpoint outputs/best_model.pth --input data/test

    # Preprocessing only
    python main.py --mode preprocess --config configs/default.yaml --input data/raw --output data/processed

    # Analysis only
    python main.py --mode analysis --config configs/default.yaml --input outputs/predictions
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.utils.logger import setup_logger, get_logger
from src.utils.seed import set_seed
from src.utils.io import load_config


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Multi-Modal Medical Image Segmentation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default config
    python main.py --mode train

    # Train with custom config and experiment name
    python main.py --mode train --config configs/experiment.yaml --exp-name my_experiment

    # Evaluate on test set
    python main.py --mode eval --checkpoint outputs/best.pth

    # Run inference on new data
    python main.py --mode inference --checkpoint outputs/best.pth --input /path/to/data

    # Generate analysis report
    python main.py --mode analysis --input outputs/predictions --output reports/
        """
    )

    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["train", "eval", "inference", "preprocess", "analysis"],
        help="Pipeline mode to run"
    )

    # Configuration
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to configuration file (default: configs/default.yaml)"
    )

    # Experiment settings
    parser.add_argument(
        "--exp-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (overrides config)"
    )

    # Data paths
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Input data path (for inference/preprocess/analysis)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path (for inference/preprocess/analysis)"
    )

    # Model checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint (for eval/inference)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )

    # Hardware settings
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu", "mps"],
        help="Device to use (overrides config)"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Number of data loading workers (overrides config)"
    )

    # Training overrides
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )

    # Model architecture overrides
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["swin_unetr", "unet", "attention_unet", "dual_encoder"],
        help="Model architecture (overrides config)"
    )
    parser.add_argument(
        "--fusion",
        type=str,
        default=None,
        choices=["early", "late", "attention", "cross_attention"],
        help="Fusion strategy (overrides config)"
    )

    # Modality settings
    parser.add_argument(
        "--modalities",
        type=str,
        nargs="+",
        default=None,
        help="Input modalities to use, e.g., --modalities CT PET"
    )

    # Analysis options
    parser.add_argument(
        "--suv-analysis",
        action="store_true",
        help="Run SUV analysis (analysis mode)"
    )
    parser.add_argument(
        "--tmtv-analysis",
        action="store_true",
        help="Run TMTV analysis (analysis mode)"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Generate histograms (analysis mode)"
    )
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate analysis report (analysis mode)"
    )

    # Explainability options
    parser.add_argument(
        "--gradcam",
        action="store_true",
        help="Generate GradCAM visualizations"
    )
    parser.add_argument(
        "--attention-maps",
        action="store_true",
        help="Visualize attention maps"
    )
    parser.add_argument(
        "--tsne",
        action="store_true",
        help="Generate t-SNE visualization"
    )

    # Miscellaneous
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )

    return parser.parse_args()


def merge_config_with_args(config: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    """Merge command line arguments into configuration."""
    # Experiment settings
    if args.exp_name is not None:
        config["experiment"]["name"] = args.exp_name
    if args.output_dir is not None:
        config["experiment"]["output_dir"] = args.output_dir
    if args.seed is not None:
        config["experiment"]["seed"] = args.seed

    # Hardware settings
    if args.device is not None:
        config["hardware"]["device"] = args.device
    if args.num_workers is not None:
        config["hardware"]["num_workers"] = args.num_workers

    # Training settings
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["optimizer"]["lr"] = args.lr

    # Model settings
    if args.model is not None:
        config["model"]["name"] = args.model
    if args.fusion is not None:
        config["model"]["fusion"]["type"] = args.fusion

    # Modality settings
    if args.modalities is not None:
        config["data"]["modalities"] = args.modalities

    # Analysis settings
    if args.suv_analysis:
        config["analysis"]["suv"]["enabled"] = True
    if args.tmtv_analysis:
        config["analysis"]["tmtv"]["enabled"] = True
    if args.histogram:
        config["analysis"]["histogram"]["enabled"] = True

    # Explainability settings
    if args.gradcam:
        config["explainability"]["gradcam"]["enabled"] = True
    if args.attention_maps:
        config["explainability"]["attention_maps"]["enabled"] = True
    if args.tsne:
        config["explainability"]["tsne"]["enabled"] = True

    # Store args for reference
    config["_args"] = {
        "mode": args.mode,
        "input": args.input,
        "output": args.output,
        "checkpoint": args.checkpoint,
        "resume": args.resume,
        "verbose": args.verbose,
        "debug": args.debug,
        "generate_report": args.generate_report,
    }

    return config


def run_train(config: Dict[str, Any], logger) -> None:
    """Run training pipeline."""
    from src.data import get_dataloader
    from src.models import build_model
    from src.trainer import Trainer

    logger.info("Starting training pipeline")
    logger.info(f"Experiment: {config['experiment']['name']}")

    # Build data loaders
    train_loader = get_dataloader(config, split="train")
    val_loader = get_dataloader(config, split="val")

    # Build model
    model = build_model(config)

    # Initialize trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        logger=logger,
        resume_from=config["_args"].get("resume"),
    )

    # Run training
    trainer.train()

    logger.info("Training completed")


def run_eval(config: Dict[str, Any], logger) -> None:
    """Run evaluation pipeline."""
    from src.data import get_dataloader
    from src.models import build_model
    from src.trainer import Trainer

    checkpoint_path = config["_args"].get("checkpoint")
    if checkpoint_path is None:
        raise ValueError("--checkpoint is required for evaluation mode")

    logger.info("Starting evaluation pipeline")
    logger.info(f"Checkpoint: {checkpoint_path}")

    # Build data loader
    test_loader = get_dataloader(config, split="test")

    # Build and load model
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize trainer and evaluate
    trainer = Trainer(
        config=config,
        model=model,
        val_loader=test_loader,
        logger=logger,
    )

    metrics = trainer.evaluate()

    logger.info("Evaluation completed")
    logger.info(f"Results: {metrics}")


def run_inference(config: Dict[str, Any], logger) -> None:
    """Run inference pipeline."""
    from src.models import build_model
    from src.trainer import Trainer

    checkpoint_path = config["_args"].get("checkpoint")
    input_path = config["_args"].get("input")
    output_path = config["_args"].get("output", "outputs/predictions")

    if checkpoint_path is None:
        raise ValueError("--checkpoint is required for inference mode")
    if input_path is None:
        raise ValueError("--input is required for inference mode")

    logger.info("Starting inference pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    # Build and load model
    model = build_model(config)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    # Initialize trainer and run inference
    trainer = Trainer(
        config=config,
        model=model,
        logger=logger,
    )

    trainer.predict(input_path=input_path, output_path=output_path)

    logger.info("Inference completed")


def run_preprocess(config: Dict[str, Any], logger) -> None:
    """Run preprocessing pipeline."""
    from src.preprocessing import DicomConverter, SUVCalculator, ImageRegistration

    input_path = config["_args"].get("input")
    output_path = config["_args"].get("output", "data/processed")

    if input_path is None:
        raise ValueError("--input is required for preprocess mode")

    logger.info("Starting preprocessing pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    # Initialize preprocessors
    converter = DicomConverter(config)
    suv_calc = SUVCalculator(config)
    registrator = ImageRegistration(config)

    # Process each modality
    for modality in config["data"]["modalities"]:
        modality_input = os.path.join(input_path, modality)
        modality_output = os.path.join(output_path, modality)

        if os.path.exists(modality_input):
            logger.info(f"Processing {modality} images")

            # Convert DICOM to NIfTI
            nifti_path = converter.convert(modality_input, modality_output, modality=modality)

            # Calculate SUV for PET
            if modality == "PET":
                suv_calc.calculate(modality_input, modality_output)

    # Register images if multiple modalities
    if len(config["data"]["modalities"]) > 1 and config["data"]["registration"]["enabled"]:
        logger.info("Registering images")
        registrator.register(output_path, config["data"]["primary_modality"])

    logger.info("Preprocessing completed")


def run_analysis(config: Dict[str, Any], logger) -> None:
    """Run analysis pipeline."""
    from src.analysis import SUVAnalyzer, TMTVAnalyzer, HistogramAnalyzer, ReportGenerator

    input_path = config["_args"].get("input")
    output_path = config["_args"].get("output", "outputs/analysis")

    if input_path is None:
        raise ValueError("--input is required for analysis mode")

    logger.info("Starting analysis pipeline")
    logger.info(f"Input: {input_path}")
    logger.info(f"Output: {output_path}")

    os.makedirs(output_path, exist_ok=True)

    results = {}

    # SUV Analysis
    if config["analysis"]["suv"]["enabled"]:
        logger.info("Running SUV analysis")
        suv_analyzer = SUVAnalyzer(config)
        results["suv"] = suv_analyzer.analyze(input_path, output_path)

    # TMTV Analysis
    if config["analysis"]["tmtv"]["enabled"]:
        logger.info("Running TMTV analysis")
        tmtv_analyzer = TMTVAnalyzer(config)
        results["tmtv"] = tmtv_analyzer.analyze(input_path, output_path)

    # Histogram Analysis
    if config["analysis"]["histogram"]["enabled"]:
        logger.info("Generating histograms")
        hist_analyzer = HistogramAnalyzer(config)
        results["histogram"] = hist_analyzer.analyze(input_path, output_path)

    # Generate Report
    if config["_args"].get("generate_report", False):
        logger.info("Generating report")
        report_gen = ReportGenerator(config)
        report_gen.generate(results, output_path)

    logger.info("Analysis completed")


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Merge with command line arguments
    config = merge_config_with_args(config, args)

    # Setup logging
    log_dir = Path(config["experiment"]["log_dir"]) / config["experiment"]["name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="main",
        log_file=log_dir / f"{args.mode}.log",
        level="DEBUG" if args.debug else "INFO",
    )

    # Set random seed
    set_seed(config["experiment"]["seed"])

    # Log configuration
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {args.config}")
    if args.verbose:
        logger.debug(f"Full config: {config}")

    # Run appropriate pipeline
    try:
        if args.mode == "train":
            run_train(config, logger)
        elif args.mode == "eval":
            run_eval(config, logger)
        elif args.mode == "inference":
            run_inference(config, logger)
        elif args.mode == "preprocess":
            run_preprocess(config, logger)
        elif args.mode == "analysis":
            run_analysis(config, logger)
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
