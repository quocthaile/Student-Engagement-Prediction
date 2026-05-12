import argparse
import json
import subprocess
import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DEPLOYMENT_DIR = BASE_DIR / "deployment_models"
BEST_MODEL_METADATA_FILE = DEPLOYMENT_DIR / "best_model_metadata.json"
CONFIG_SEARCH_RESULTS_JSON = DEPLOYMENT_DIR / "config_search_results.json"

try:
    from config import (
        CONFIG_SEARCH_RESULTS_FILE as CONFIG_SEARCH_RESULTS_CSV,
        PIPELINE_AUTO_SELECT_CANDIDATES,
        PIPELINE_AUTO_SELECT_MODE,
    )
except Exception:
    CONFIG_SEARCH_RESULTS_CSV = DEPLOYMENT_DIR / "config_search_results.csv"
    PIPELINE_AUTO_SELECT_MODE = "accuracy_recall"
    PIPELINE_AUTO_SELECT_CANDIDATES = [
        "balanced_3way_minmax",
        "early_warning_robust",
        "balanced_3way_robust",
        "aggressive_low_robust",
        "natural_distribution_robust",
    ]

STEPS = [
    ("1", "stage_1_generate_ground_truth.py", "Generate full-course ground truth labels"),
    ("2", "stage_2_time_window_features.py", "Extract early time-window features"),
    ("3", "stage_3_split_and_smote.py", "Split, encode, scale, and balance training data"),
    ("4", "stage_4_model_training_eval.py", "Train, evaluate, and export deployment model"),
    ("5", "stage_5_explain_model_xai.py", "Generate expected-result summary and XAI outputs"),
]


def _jsonify(value):
    if isinstance(value, dict):
        return {k: _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _write_overrides(overrides: dict):
    path = BASE_DIR / "runtime_overrides.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(overrides, f, ensure_ascii=False, indent=2)


def _load_best_model_metadata() -> dict:
    if not BEST_MODEL_METADATA_FILE.exists():
        raise FileNotFoundError(f"Missing model metadata: {BEST_MODEL_METADATA_FILE}")
    with open(BEST_MODEL_METADATA_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def _selection_score(metadata: dict) -> tuple:
    metrics = metadata.get("validation_metrics", {}) or {}
    accuracy = float(metrics.get("Accuracy", 0) or 0)
    recall_low = float(metrics.get("Recall_Low_Engagement", 0) or 0)
    f1_macro = float(metrics.get("F1_Macro", 0) or 0)
    balanced_acc = float(metrics.get("Balanced_Accuracy", 0) or 0)

    mode = str(PIPELINE_AUTO_SELECT_MODE).strip().lower()
    if mode == "recall_first":
        return (recall_low, accuracy, f1_macro, balanced_acc)
    if mode == "balanced":
        return (f1_macro, balanced_acc, accuracy, recall_low)
    # default: accuracy first, then recall_low as tie-breaker
    return (accuracy, recall_low, balanced_acc, f1_macro)


def _candidate_configs():
    try:
        from config import get_pipeline_search_presets
        presets = get_pipeline_search_presets()
        ordered = {}
        for preset_name in PIPELINE_AUTO_SELECT_CANDIDATES:
            if preset_name in presets:
                ordered[preset_name] = presets[preset_name]
        for preset_name, preset_value in presets.items():
            if preset_name not in ordered:
                ordered[preset_name] = preset_value
        return ordered
    except Exception:
        return {
            "early_warning_robust": {
                "LABEL_PERCENTILES": (0.60, 0.85),
                "TRAIN_CLASS_RATIOS": {
                    "Low_Engagement": 4.8,
                    "Medium_Engagement": 1.8,
                    "High_Engagement": 3.5,
                },
                "USE_ENTROPY_WEIGHT_METHOD": True,
                "LAPLACE_SMOOTHING_ALPHA": 1.0,
                "SCALER_TYPE": "robust",
            }
        }


def _run_pipeline_steps(step_ids: list[str], overrides: dict) -> None:
    for sid, fname, _ in STEPS:
        if sid in step_ids:
            print(f"Running step {sid} -> {fname}")
            run_step(fname, overrides)


def auto_select_best_config(step_ids: list[str] | None = None, save_summary: bool = True) -> None:
    step_ids = step_ids or ["1", "2", "3", "4"]
    presets = _candidate_configs()
    results = []
    best = None

    print(f"Auto-selecting config preset using mode: {PIPELINE_AUTO_SELECT_MODE}")
    for preset_name, preset_overrides in presets.items():
        print("-" * 80)
        print(f"Testing preset: {preset_name}")
        _run_pipeline_steps(step_ids, preset_overrides)
        metadata = _load_best_model_metadata()
        score = _selection_score(metadata)
        validation_metrics = metadata.get("validation_metrics", {}) or {}
        test_metrics = metadata.get("test_metrics", {}) or {}
        record = {
            "preset_name": preset_name,
            "score": score,
            "selection_mode": PIPELINE_AUTO_SELECT_MODE,
            "model_name": metadata.get("model_name"),
            "model_class": metadata.get("model_class"),
            "validation_accuracy": validation_metrics.get("Accuracy"),
            "validation_recall_low": validation_metrics.get("Recall_Low_Engagement"),
            "validation_f1_macro": validation_metrics.get("F1_Macro"),
            "validation_balanced_accuracy": validation_metrics.get("Balanced_Accuracy"),
            "test_accuracy": test_metrics.get("Accuracy"),
            "test_recall_low": test_metrics.get("Recall_Low_Engagement"),
            "test_f1_macro": test_metrics.get("F1_Macro"),
            "test_balanced_accuracy": test_metrics.get("Balanced_Accuracy"),
            "validation_metrics": validation_metrics,
            "test_metrics": test_metrics,
            "overrides": _jsonify(preset_overrides),
        }
        results.append(record)
        if best is None or score > best["score"]:
            best = record
        print(f"Preset {preset_name} score = {score}")

    if best is None:
        raise RuntimeError("No configuration preset produced a valid result")

    print("-" * 80)
    comparison_df = pd.DataFrame(results)
    sort_cols = ["validation_accuracy", "validation_recall_low", "validation_f1_macro", "validation_balanced_accuracy"]
    comparison_df = comparison_df.sort_values(by=sort_cols, ascending=False, na_position="last")
    print("FULL CONFIG COMPARISON TABLE")
    print(comparison_df[[
        "preset_name",
        "model_name",
        "selection_mode",
        "validation_accuracy",
        "validation_recall_low",
        "validation_f1_macro",
        "validation_balanced_accuracy",
        "test_accuracy",
        "test_recall_low",
    ]].to_string(index=False))

    best = comparison_df.iloc[0].to_dict()
    print(f"Best preset: {best['preset_name']} with Accuracy={best.get('validation_accuracy')} and Recall_Low={best.get('validation_recall_low')}")
    print("Re-running best preset so final artifacts match the selected configuration...")
    _run_pipeline_steps(step_ids, _candidate_configs()[best["preset_name"]])

    if save_summary:
        summary = {
            "selection_mode": PIPELINE_AUTO_SELECT_MODE,
            "best_preset": best,
            "all_results": results,
        }
        CONFIG_SEARCH_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_SEARCH_RESULTS_JSON, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, default=str)
        print(f"Saved config search summary to: {CONFIG_SEARCH_RESULTS_JSON}")
        CONFIG_SEARCH_RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
        comparison_df.to_csv(CONFIG_SEARCH_RESULTS_CSV, index=False, encoding="utf-8-sig")
        print(f"Saved config comparison CSV to: {CONFIG_SEARCH_RESULTS_CSV}")


def run_step(script_name: str, env_overrides: dict | None = None) -> None:
    script_path = BASE_DIR / script_name
    env = None
    if env_overrides:
        _write_overrides(env_overrides)
    cmd = [sys.executable, str(script_path)]
    step_name = next((desc for sid, fname, desc in STEPS if fname == script_name), script_name)
    print(f"Running: {step_name}")
    print(f"Executing command: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=BASE_DIR)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    if script_name == "stage_4_model_training_eval.py":
        print_stage_4_summary()


def print_stage_4_summary() -> None:
    if not BEST_MODEL_METADATA_FILE.exists():
        print(f"Stage 4 summary skipped: missing {BEST_MODEL_METADATA_FILE}")
        return

    metadata = _load_best_model_metadata()
    validation_metrics = metadata.get("validation_metrics", {}) or {}
    test_metrics = metadata.get("test_metrics", {}) or {}
    selection_policy = metadata.get("selection_policy", {}) or {}

    print("-" * 80)
    print("STAGE 4 SELECTION SUMMARY")
    print(f"Selected model: {metadata.get('model_name')} ({metadata.get('model_class')})")
    print(f"Selection split: {selection_policy.get('source_split', 'valid')}")
    print(f"Decision policy: {selection_policy.get('decision_policy')}")
    print(f"Low threshold: {selection_policy.get('low_threshold')}")
    print("Validation metrics used for selection:")
    print(
        "  "
        f"Recall_Low={validation_metrics.get('Recall_Low_Engagement')}, "
        f"Precision_Low={validation_metrics.get('Precision_Low_Engagement')}, "
        f"F1_Low={validation_metrics.get('F1_Low_Engagement')}, "
        f"AUC_ROC_OVR={validation_metrics.get('AUC_ROC_OVR')}, "
        f"F1_Macro={validation_metrics.get('F1_Macro')}, "
        f"Balanced_Accuracy={validation_metrics.get('Balanced_Accuracy')}, "
        f"Accuracy={validation_metrics.get('Accuracy')}"
    )
    print("Final held-out test metrics:")
    print(
        "  "
        f"Recall_Low={test_metrics.get('Recall_Low_Engagement')}, "
        f"Precision_Low={test_metrics.get('Precision_Low_Engagement')}, "
        f"F1_Low={test_metrics.get('F1_Low_Engagement')}, "
        f"AUC_ROC_OVR={test_metrics.get('AUC_ROC_OVR')}, "
        f"F1_Macro={test_metrics.get('F1_Macro')}, "
        f"Balanced_Accuracy={test_metrics.get('Balanced_Accuracy')}, "
        f"Accuracy={test_metrics.get('Accuracy')}, "
        f"Decision_Policy={selection_policy.get('decision_policy')}"
    )
    print(f"Validation comparison CSV: {DEPLOYMENT_DIR / 'evaluation_metrics.csv'}")
    print(f"Final test metrics CSV: {DEPLOYMENT_DIR / 'final_test_metrics.csv'}")
    print(f"Detailed stage 4 log: {DEPLOYMENT_DIR / 'stage_4_training_eval.log'}")
    print("-" * 80)


def interactive_menu():
    print("Simple interactive pipeline manager")
    overrides = {}
    while True:
        print("\nAvailable steps:")
        for sid, fname, desc in STEPS:
            print(f" {sid}. {desc}    ({fname})")
        print(" a. Run all steps")
        print(" r. Run a range (e.g. 1-3)")
        print(" p. Set pipeline parameter override (KEY=JSON_VALUE)")
        print(" s. Show current overrides")
        print(" q. Quit")
        choice = input("Select action: ").strip()
        if choice == "q":
            return
        if choice == "a":
            for sid, fname, _ in STEPS:
                print(f"Running step {sid} -> {fname}")
                run_step(fname, overrides)
            continue
        if choice == "r":
            rng = input("Enter range from-to (e.g. 1-3): ").strip()
            if "-" not in rng:
                print("Invalid range")
                continue
            a, b = rng.split("-", 1)
            try:
                a = int(a); b = int(b)
            except Exception:
                print("Invalid numbers")
                continue
            for sid, fname, _ in STEPS:
                if a <= int(sid) <= b:
                    print(f"Running step {sid} -> {fname}")
                    run_step(fname, overrides)
            continue
        if choice == "p":
            pair = input("Enter KEY=JSON_VALUE (e.g. TRAIN_TARGET_TOTAL_SAMPLES=60000): ").strip()
            if "=" not in pair:
                print("Invalid input")
                continue
            k, v = pair.split("=", 1)
            k = k.strip()
            try:
                parsed = json.loads(v)
            except Exception:
                parsed = v
            overrides[k] = parsed
            print(f"Set override {k} -> {parsed}")
            continue
        if choice == "s":
            print(json.dumps(overrides, indent=2, ensure_ascii=False))
            continue
        if choice.isdigit():
            found = False
            for sid, fname, _ in STEPS:
                if sid == choice:
                    found = True
                    run_step(fname, overrides)
            if not found:
                print("Unknown step")
            continue
        print("Unknown option")


def main():
    parser = argparse.ArgumentParser(description="Menu-driven pipeline manager")
    parser.add_argument("--from-step", help="First step to run", choices=[s[0] for s in STEPS])
    parser.add_argument("--to-step", help="Last step to run", choices=[s[0] for s in STEPS])
    parser.add_argument("--param", help="Override param as KEY=JSON (can be repeated)", action="append")
    parser.add_argument("--auto-select-config", action="store_true", help="Run preset search and choose the best config")
    parser.add_argument("--menu", help="Interactive menu", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.param:
        for p in args.param:
            if "=" not in p:
                continue
            k, v = p.split("=", 1)
            try:
                overrides[k] = json.loads(v)
            except Exception:
                overrides[k] = v

    if args.menu:
        interactive_menu()
        return

    if args.auto_select_config:
        auto_select_best_config(step_ids=["1", "2", "3", "4"])
        return

    if args.from_step and args.to_step:
        a = int(args.from_step); b = int(args.to_step)
        for sid, fname, _ in STEPS:
            if a <= int(sid) <= b:
                print(f"Running step {sid} -> {fname}")
                run_step(fname, overrides)
        return

    for sid, fname, _ in STEPS:
        print(f"Running step {sid} -> {fname}")
        run_step(fname, overrides)


if __name__ == "__main__":
    main()
