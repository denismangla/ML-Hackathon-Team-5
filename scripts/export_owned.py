"""Copy a package's reviewed, explicitly listed files into a teammate's git checkout."""
import argparse
import hashlib
from pathlib import Path
import shutil

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('checkout',type=Path)
    target=parser.parse_args().checkout.expanduser().resolve()
    root=Path(__file__).resolve().parents[1]
    if target==root or not (target/'.git').exists():
        parser.error('Choose a separate local Git checkout.')
    manifest=root/'OWNED_FILES.txt'
    if not manifest.exists():
        parser.error('Run the export from an extracted individual handoff ZIP.')
    entries=[Path(line) for line in manifest.read_text().splitlines() if line]
    todo=[]
    for rel in entries:
        src=root/rel
        dst=target/rel
        if src.resolve().is_relative_to(root) and dst.resolve().is_relative_to(target) and src.is_file():
            if dst.exists():
                if not dst.is_file() or hashlib.sha256(dst.read_bytes()).digest()!=hashlib.sha256(src.read_bytes()).digest():
                    parser.error(f'Existing change in {rel}; reconcile it in the checkout first.')
            else:
                todo.append((src,dst))
        else:
            parser.error(f'Invalid manifest entry: {rel}')
    for src,dst in todo:
        dst.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(src,dst)
    print(f'Copied {len(todo)} assigned files to {target}. Review git status and commit your work.')

if __name__=='__main__':
    main()
