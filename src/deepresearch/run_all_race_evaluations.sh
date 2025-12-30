#!/bin/bash
# Run RACE evaluation for all 5 runs of the 20×5 experiment

set -e  # Exit on error

# Check if experiment name is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <experiment_name>"
    echo "Example: $0 jul19_llama405b_llama1b"
    exit 1
fi

EXPERIMENT_NAME=$1

echo "🏁 Starting RACE evaluation for all 5 runs of $EXPERIMENT_NAME"
echo "================================================="

cd deep_research_bench

# Loop through all 5 runs
for i in {1..5}; do
    echo ""
    echo "📊 Running RACE evaluation for Run $i..."
    echo "------------------------------------------------"
    
    python deepresearch_bench_race.py ${EXPERIMENT_NAME}_run$i \
        --raw_data_dir data/test_data/raw_data \
        --output_dir results/race/${EXPERIMENT_NAME}_run$i \
        --max_workers 8
    
    echo "✅ Run $i evaluation complete"
done

echo ""
echo "🎉 All RACE evaluations complete!"
echo "================================================="
echo ""
echo "📁 Results saved in:"
for i in {1..5}; do
    echo "   - deep_research_bench/results/race/${EXPERIMENT_NAME}_run$i/"
done