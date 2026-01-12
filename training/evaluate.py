"""Evaluate full pipeline on all labeled data."""

from pathlib import Path

import pandas as pd
from labor_union_parser import Extractor

SCRIPT_DIR = Path(__file__).parent


def main():
    # Load data
    df = pd.read_csv(SCRIPT_DIR / "data/labeled_data.csv")
    print(f"Total examples: {len(df)}")

    # Filter to test split only
    test_df = df[df["split"] == "test"].copy()
    print(f"Test split examples: {len(test_df)}")

    # Filter to known affiliations
    extractor = Extractor()
    known_df = test_df[test_df["aff_abbr"].isin(extractor.aff_list)].copy()
    print(f"Test examples with known affiliations: {len(known_df)}")

    # Run extraction
    texts = known_df["text"].tolist()
    results = list(extractor.extract_all(texts, batch_size=256, show_progress=True))

    # Evaluate
    aff_correct = 0
    aff_non_none_total = 0
    aff_non_none_correct = 0
    desig_correct = 0
    desig_correct_given_has_desig = 0
    desig_correct_given_no_desig = 0
    joint_correct = 0
    joint_non_none_correct = 0

    has_desig_total = 0
    no_desig_total = 0

    errors = []

    for i, (idx, row) in enumerate(known_df.iterrows()):
        result = results[i]
        true_aff = row["aff_abbr"]
        true_desig = (
            str(row["desig_num"]).split(".")[0] if pd.notna(row["desig_num"]) else ""
        )
        true_desig = true_desig.lstrip("0") or ""

        pred_aff = result["affiliation"]
        pred_desig = result["designation"].lstrip("0") if result["designation"] else ""

        aff_match = pred_aff == true_aff
        desig_match = pred_desig == true_desig

        if aff_match:
            aff_correct += 1

        # Track non-None predictions (confident predictions)
        if pred_aff is not None:
            aff_non_none_total += 1
            if aff_match:
                aff_non_none_correct += 1
            if aff_match and desig_match:
                joint_non_none_correct += 1

        if desig_match:
            desig_correct += 1

        if true_desig:
            has_desig_total += 1
            if desig_match:
                desig_correct_given_has_desig += 1
        else:
            no_desig_total += 1
            if desig_match:
                desig_correct_given_no_desig += 1

        if aff_match and desig_match:
            joint_correct += 1
        else:
            errors.append(
                {
                    "text": row["text"][:80],
                    "true_aff": true_aff,
                    "pred_aff": pred_aff,
                    "aff_match": aff_match,
                    "true_desig": true_desig,
                    "pred_desig": pred_desig,
                    "desig_match": desig_match,
                }
            )

    total = len(known_df)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"\nAffiliation accuracy:  {aff_correct}/{total} = {aff_correct/total:.4f}")
    print(
        f"  - Non-None preds:    {aff_non_none_correct}/{aff_non_none_total} = {aff_non_none_correct/aff_non_none_total:.4f}"
    )
    print(f"Designation accuracy:  {desig_correct}/{total} = {desig_correct/total:.4f}")
    print(
        f"  - With desig:        {desig_correct_given_has_desig}/{has_desig_total} = {desig_correct_given_has_desig/has_desig_total:.4f}"
    )
    print(
        f"  - Without desig:     {desig_correct_given_no_desig}/{no_desig_total} = {desig_correct_given_no_desig/no_desig_total:.4f}"
    )
    print(f"Joint accuracy:        {joint_correct}/{total} = {joint_correct/total:.4f}")
    print(
        f"  - Non-None preds:    {joint_non_none_correct}/{aff_non_none_total} = {joint_non_none_correct/aff_non_none_total:.4f}"
    )

    # Save errors
    errors_df = pd.DataFrame(errors)
    errors_df.to_csv(SCRIPT_DIR / "data/pipeline_errors.csv", index=False)
    print(f"\nSaved {len(errors)} errors to training/data/pipeline_errors.csv")

    # Summary by error type
    aff_only_errors = sum(1 for e in errors if not e["aff_match"] and e["desig_match"])
    desig_only_errors = sum(
        1 for e in errors if e["aff_match"] and not e["desig_match"]
    )
    both_errors = sum(1 for e in errors if not e["aff_match"] and not e["desig_match"])

    print("\nError breakdown:")
    print(f"  Affiliation only:    {aff_only_errors}")
    print(f"  Designation only:    {desig_only_errors}")
    print(f"  Both wrong:          {both_errors}")


if __name__ == "__main__":
    main()
