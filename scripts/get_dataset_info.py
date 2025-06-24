import os
import sys
import numpy as np
import argparse
from pathlib import Path

def print_data_structure(data, prefix=""):
    """Print data structure showing only shape and type, recursively for dicts."""
    if isinstance(data, dict):
        for key in sorted(data.keys()):
            value = data[key]
            if isinstance(value, np.ndarray):
                print(f"{prefix}{key} (Dataset)")
                print(f"{prefix}  - Type: {value.dtype}")
                print(f"{prefix}  - Shape: {value.shape}")
            elif isinstance(value, dict):
                print(f"{prefix}{key}/ (Group)")
                print(f"{prefix}  - Type: Group")
                print(f"{prefix}  - Keys: {list(value.keys())}")
                print_data_structure(value, prefix + "  ")
            elif isinstance(value, (list, tuple)):
                print(f"{prefix}{key} (List/Tuple)")
                print(f"{prefix}  - Type: {type(value).__name__}")
                print(f"{prefix}  - Length: {len(value)}")
                if len(value) > 0:
                    elem = value[0]
                    if hasattr(elem, 'shape'):
                        print(f"{prefix}  - Element shape: {elem.shape}")
                    else:
                        print(f"{prefix}  - Element type: {type(elem).__name__}")
            else:
                t = type(value).__name__
                try:
                    length = len(value)
                    print(f"{prefix}{key} (Other)")
                    print(f"{prefix}  - Type: {t}")
                    print(f"{prefix}  - Length: {length}")
                except Exception:
                    print(f"{prefix}{key} (Value)")
                    print(f"{prefix}  - Type: {t}")

def inspect_specific_episode(ep_path):
    """Load a single .npz episode file and print its internal structure."""
    if not os.path.isfile(ep_path):
        print(f"Error: File does not exist: {ep_path}", file=sys.stderr)
        return
    try:
        npz = np.load(ep_path, allow_pickle=True)
    except Exception as e:
        print(f"!!! Unable to load {ep_path}: {e}", file=sys.stderr)
        return

    # convert NpzFile to plain dict for ease of recursion
    if isinstance(npz, np.lib.npyio.NpzFile):
        data = {k: npz[k] for k in npz.files}
    else:
        # in case it's a pickled dict
        try:
            data = npz.item()
        except:
            print(f"!!! Unsupported file format: {ep_path}", file=sys.stderr)
            return

    print(f"\n=== SPECIFIC EPISODE DETAIL ===")
    print(f"Episode: {ep_path}")
    print("-" * (10 + len(ep_path)))
    print_data_structure(data)

def analyze_single_episode(episode_path):
    """Analyze detailed information of a single episode file (returns dict)."""
    try:
        raw = np.load(episode_path, allow_pickle=True)
        if isinstance(raw, np.lib.npyio.NpzFile):
            data = {k: raw[k] for k in raw.files}
        else:
            data = raw.item()
        return {
            'file_path': episode_path,
            'file_size_mb': os.path.getsize(episode_path) / (1024 * 1024),
            'data': data,
        }
    except Exception as e:
        return {'error': str(e), 'file_path': episode_path}

def get_dataset_info(dataset_path):
    """Get and print summary information for training/validation splits."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}", file=sys.stderr)
        return

    print("="*80)
    print(f"CALVIN Dataset Analysis")
    print(f"Dataset Path: {dataset_path}")
    print("="*80)

    for split in ['training', 'validation']:
        split_path = dataset_path / split
        if not split_path.exists():
            print(f"\nWarning: {split} directory does not exist", file=sys.stderr)
            continue

        print(f"\n{split.upper()} DATA ANALYSIS:")
        print("-"*50)
        episode_files = sorted(split_path.glob('episode_*.npz'))
        print(f"Number of episodes in {split}: {len(episode_files)}")

        if episode_files:
            # Only print the structure summary of the first episode
            info = analyze_single_episode(str(episode_files[0]))
            if 'data' in info and isinstance(info['data'], dict):
                print(f"\nFile: {info['file_path']}")
                print("-"*len(str(episode_files[0])))
                print_data_structure(info['data'])

        # Print language annotation structure (if exists)
        lang_dir = split_path / 'lang_annotations'
        if lang_dir.exists():
            print(f"\nLANGUAGE ANNOTATIONS ({split}):")
            print("-"*50)
            auto_path = lang_dir / 'auto_lang_ann.npy'
            if auto_path.exists():
                try:
                    ann = np.load(auto_path, allow_pickle=True).item()
                    print(f"\nFile: {auto_path}")
                    print("-"*len(str(auto_path)))
                    print_data_structure(ann)
                except Exception as e:
                    print(f"Error loading {auto_path}: {e}", file=sys.stderr)
            emb_path = lang_dir / 'embeddings.npy'
            if emb_path.exists():
                try:
                    emb = np.load(emb_path, allow_pickle=True)
                    print(f"\nFile: {emb_path}")
                    print("-"*len(str(emb_path)))
                    print("embeddings (Dataset)")
                    print(f"  - Type: {emb.dtype}")
                    print(f"  - Shape: {emb.shape}")
                except Exception as e:
                    print(f"Error loading {emb_path}: {e}", file=sys.stderr)

def main():
    parser = argparse.ArgumentParser(description='Analyze CALVIN dataset structure')
    parser.add_argument('dataset_path', help='Dataset root directory path')
    parser.add_argument(
        '--episode', '-e',
        help='(Optional) Specify an episode_*.npz file path to print detailed structure of that file'
    )
    args = parser.parse_args()

    # Print training/validation overview
    get_dataset_info(args.dataset_path)

    # If --episode is passed, print detailed information of that file separately
    if args.episode:
        inspect_specific_episode(args.episode)

if __name__ == "__main__":
    main()