#!/usr/bin/env python3
"""
Main CLI entry point for ResSwarm experiments.

This script provides a unified interface for running ResSwarm experiments
with the proper path setup and error handling.
"""

import sys
import os
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "res_swarm"))
sys.path.insert(0, str(project_root / "minions"))

def main():
    """Main entry point that delegates to the actual runner."""
    try:
        # Import here to ensure paths are set up first
        from res_swarm.src.deepres_runner import main as runner_main
        
        # Run the main deepres runner
        runner_main()
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("\nTroubleshooting:")
        print("1. Ensure you're in the project root directory")
        print("2. Run: python res_swarm/scripts/setup_deepres.py")
        print("3. Check that minions/ submodule is properly set up")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()