#!/usr/bin/env python3
"""
Build training dataset from validated pairs.

Creates train/val splits from validated Cypher pairs.
Also generates category/difficulty statistics for balanced allocation.

Example:
    python scripts/build_dataset.py \\
        --input data/validated/validated_pairs.parquet \\
        --output data/splits/ \\
        --train-split 0.8 \\
        --val-split 0.1
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).parent.parent))


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Analyze dataset distribution."""
    stats = {
        "total_pairs": len(df),
        "unique_questions": df["question"].nunique(),
        "avg_question_len": df["question"].str.len().mean(),
        "avg_cypher_len": df["cypher"].str.len().mean(),
    }
    
    if "category" in df.columns:
        stats["categories"] = df["category"].value_counts().to_dict()
    
    if "difficulty" in df.columns:
        stats["difficulty"] = df["difficulty"].value_counts().to_dict()
    
    return stats


def build_splits(
    input_path: str,
    output_dir: str,
    train_split: float = 0.8,
    val_split: float = 0.1,
    stratify: str = None,
):
    """
    Build train/val/test splits from validated dataset.
    
    Args:
        input_path: Path to validated parquet file
        output_dir: Output directory for splits
        train_split: Fraction for training (default 0.8)
        val_split: Fraction for validation (default 0.1)
        stratify: Column to stratify by (category, difficulty, etc.)
    """
    print("=" * 70)
    print("📊 Building Training Dataset")
    print("=" * 70)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Load validated pairs
    print(f"📥 Loading validated pairs: {input_path}...")
    if not Path(input_path).exists():
        print(f"❌ File not found: {input_path}")
        sys.exit(1)
    
    df = pd.read_parquet(input_path)
    print(f"   ✓ {len(df)} validated pairs loaded")
    
    # Analyze
    print(f"\n📊 Dataset analysis:")
    stats = analyze_dataset(df)
    for key, value in stats.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"     {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Check splits sum to ~1.0
    test_split = 1.0 - train_split - val_split
    if test_split < 0.01:
        print(f"\n⚠️  Warning: test split too small ({test_split:.2%}), will be minimal")
    
    print(f"\n🔀 Split ratio: {train_split:.0%} train, {val_split:.0%} val, {test_split:.0%} test")
    
    # Shuffle
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Stratify if requested
    if stratify and stratify in df.columns:
        print(f"   Stratifying by '{stratify}'...")
        # For multi-class stratification, use sklearn
        X = df.drop(columns=["question", "cypher"])
        y = df[stratify]
        
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_split,
            stratify=y,
            random_state=42,
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_split / (train_split + val_split),
            stratify=y_temp,
            random_state=42,
        )
        
        train_df = pd.concat([X_train.reset_index(drop=True), 
                             df.loc[X_train.index, ["question", "cypher"]].reset_index(drop=True)], axis=1)
        val_df = pd.concat([X_val.reset_index(drop=True),
                           df.loc[X_val.index, ["question", "cypher"]].reset_index(drop=True)], axis=1)
        test_df = pd.concat([X_test.reset_index(drop=True),
                            df.loc[X_test.index, ["question", "cypher"]].reset_index(drop=True)], axis=1)
    else:
        # Simple split
        n = len(df)
        train_end = int(n * train_split)
        val_end = train_end + int(n * val_split)
        
        train_df = df[:train_end].reset_index(drop=True)
        val_df = df[train_end:val_end].reset_index(drop=True)
        test_df = df[val_end:].reset_index(drop=True)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print(f"\n💾 Saving splits to {output_dir}...")
    
    train_path = output_dir / "train.parquet"
    val_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    val_df.to_parquet(val_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"   ✓ Train: {train_path} ({len(train_df)} pairs)")
    print(f"   ✓ Val:   {val_path} ({len(val_df)} pairs)")
    print(f"   ✓ Test:  {test_path} ({len(test_df)} pairs)")
    
    # Verify splits
    print(f"\n✅ Splits verified:")
    print(f"   Train: {len(train_df):6d} ({len(train_df)/len(df)*100:5.1f}%)")
    print(f"   Val:   {len(val_df):6d} ({len(val_df)/len(df)*100:5.1f}%)")
    print(f"   Test:  {len(test_df):6d} ({len(test_df)/len(df)*100:5.1f}%)")
    print(f"   Total: {len(df):6d}")
    
    return train_df, val_df, test_df


def create_category_budget(df: pd.DataFrame, target_total: int = 10000):
    """
    Generate a balanced category budget for data generation.
    
    Allocates based on CLAUDE.md's weak-5 strategy.
    """
    print(f"\n📋 Category budget for next generation round (target {target_total} total):")
    
    # Default allocation from CLAUDE.md
    budget = {
        "weak_5_categories": 0.45,
        "other_medium_hard": 0.30,
        "simple_retrieval": 0.15,
        "edge_cases": 0.10,
    }
    
    print(f"   Allocation (from CLAUDE.md):")
    for bucket, fraction in budget.items():
        count = int(target_total * fraction)
        print(f"     {bucket:25s}: {count:5d} ({fraction*100:5.1f}%)")
    
    return budget


def main():
    parser = argparse.ArgumentParser(description="Build training dataset from validated pairs")
    parser.add_argument("--input", default="data/validated/validated_pairs.parquet",
                        help="Validated pairs (parquet)")
    parser.add_argument("--output", default="data/splits/",
                        help="Output directory for splits")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Fraction for training")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Fraction for validation")
    parser.add_argument("--stratify", default=None,
                        help="Stratify by column (category, difficulty)")
    parser.add_argument("--show-budget", action="store_true",
                        help="Show recommended generation budget")
    
    args = parser.parse_args()
    
    # Build splits
    train_df, val_df, test_df = build_splits(
        args.input,
        args.output,
        args.train_split,
        args.val_split,
        args.stratify,
    )
    
    # Show budget if requested
    if args.show_budget:
        create_category_budget(train_df, target_total=10000)
    
    print("\n✅ Dataset ready for training!")
    print(f"\nNext step:")
    print(f"  python scripts/train_lora.py \\")
    print(f"    --train-data {args.output}/train.parquet \\")
    print(f"    --val-data {args.output}/val.parquet")


if __name__ == "__main__":
    main()
