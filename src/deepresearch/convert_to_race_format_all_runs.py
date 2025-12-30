#!/usr/bin/env python3
"""
Convert 20×5 experiment results to RACE evaluation format.
Creates separate JSONL files for each of the 5 runs.
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict

def convert_to_race_format(experiment_name):
    """Convert raw experiment results to RACE evaluation format."""
    
    # Paths
    raw_results_dir = Path(f"outputs/experiments/{experiment_name}/raw_results")
    output_dir = Path("deep_research_bench/data/test_data/raw_data")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Group results by run number
    runs_data = defaultdict(list)
    
    # Expected task IDs
    expected_tasks = [54, 55, 57, 59, 63, 64, 66, 68, 70, 76, 77, 81, 83, 88, 90, 92, 96, 97, 98, 99]
    
    print("🔄 Converting 20×5 experiment results to RACE evaluation format...")
    print(f"   Source: {raw_results_dir}")
    print(f"   Target: {output_dir}")
    print()
    
    # Process all result files
    processed_files = 0
    for task_id in expected_tasks:
        for run_num in range(1, 6):  # 1 to 5
            filename = f"task_{task_id}_run_{run_num}.json"
            filepath = raw_results_dir / filename
            
            if not filepath.exists():
                print(f"⚠️  Warning: Missing file {filename}")
                continue
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract required fields for RACE evaluation
                race_entry = {
                    "id": data["task_id"],
                    "prompt": data["task_prompt"],
                    "article": data["final_answer"]
                }
                
                runs_data[run_num].append(race_entry)
                processed_files += 1
                
            except Exception as e:
                print(f"❌ Error processing {filename}: {e}")
    
    print(f"✅ Processed {processed_files} files")
    print()
    
    # Write separate JSONL files for each run
    for run_num in range(1, 6):
        if run_num not in runs_data:
            print(f"⚠️  No data found for run {run_num}")
            continue
        
        # Sort by task ID for consistent ordering
        runs_data[run_num].sort(key=lambda x: x["id"])
        
        output_filename = f"{experiment_name}_run{run_num}.jsonl"
        output_path = output_dir / output_filename
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for entry in runs_data[run_num]:
                json.dump(entry, f, ensure_ascii=False)
                f.write('\n')
        
        print(f"✅ Created {output_filename} with {len(runs_data[run_num])} tasks")
    
    print()
    print("🎉 Conversion complete!")
    print()
    print("📋 Next steps:")
    print("1. cd deep_research_bench")
    print("2. Run RACE evaluation for each file:")
    for run_num in range(1, 6):
        print(f"   python deepresearch_bench_race.py {experiment_name}_run{run_num} \\")
        print(f"     --raw_data_dir data/test_data/raw_data --output_dir results/race")
    
    return runs_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert experiment results to RACE format')
    parser.add_argument('--experiment-name', type=str, required=True,
                        help='Name of the experiment (e.g., jul19_llama405b_llama1b)')
    args = parser.parse_args()
    
    runs_data = convert_to_race_format(args.experiment_name)
    
    # Print summary statistics
    print()
    print("📊 Summary:")
    for run_num, entries in runs_data.items():
        print(f"   Run {run_num}: {len(entries)} tasks")
    
    total_entries = sum(len(entries) for entries in runs_data.values())
    print(f"   Total: {total_entries} entries across {len(runs_data)} runs")