"""Translate raw prediction files for benchmark tasks into format expected by
benchmark leaderboards.
"""
import os
import argparse

from jiant.scripts.benchmarks.benchmarks import GlueBenchmark, SuperglueBenchmark

SUPPORTED_BENCHMARKS = {"GLUE": GlueBenchmark, "SUPERGLUE": SuperglueBenchmark}


def results(benchmark: str, input_base_path: str, output_path: str, task_names: [], preds: str, regime: str):
    benchmark = SUPPORTED_BENCHMARKS[benchmark]

    for task_name in task_names:
        input_filepath = os.path.join(input_base_path, preds)
        os.makedirs(f"{output_path}/{task_name}", exist_ok=True)
        output_filepath = os.path.join(
            output_path, task_name, regime + "_" + benchmark.BENCHMARK_SUBMISSION_FILENAMES[task_name]
        )
        benchmark.write_predictions(task_name, input_filepath, output_filepath)


def main():
    parser = argparse.ArgumentParser(
        description="Generate formatted test prediction files for benchmark submission"
    )
    parser.add_argument(
        "--input_base_path",
        required=True,
        help="base input path to the folder that contain properly formatted folders of models (e.g. crosloengual__slobench__cb__epochs_10__train_batch_1__eval_batch_1__num_eval_steps_0)",
    )
    parser.add_argument("--output_path", required=True, help="output path for formatted files")
    parser.add_argument(
        "--benchmark", required=True, choices=SUPPORTED_BENCHMARKS, help="name of benchmark"
    )
    args = parser.parse_args()

    benchmark = SUPPORTED_BENCHMARKS[args.benchmark]

    # input file paths
    task_pred = {}
    for root, dirs, files in os.walk(args.input_base_path):
        if "test_preds.p" in files:
            task_name = root.split('__')[2]
            pred_file_path = os.path.join(root, "test_preds.p")
            task_pred[task_name] = pred_file_path

    for task_name, pred_file_path in task_pred.items():
        print(task_name, pred_file_path)
        input_filepath = pred_file_path
        output_filepath = os.path.join(
            args.output_path, benchmark.BENCHMARK_SUBMISSION_FILENAMES[task_name]
        )
        benchmark.write_predictions(task_name, input_filepath, output_filepath)


if __name__ == "__main__":
    main()
