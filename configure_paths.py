"""Point the AIVC analysis scripts at your own data root.

The scripts carry absolute paths from the author's workstation. This utility
rewrites them in place, keeping a .bak copy of every file it changes.

Usage:
    python configure_paths.py --root /path/to/aivc --dry-run
    python configure_paths.py --root /path/to/aivc
"""

import argparse
import os
import sys

TARGET_DIRS = ['func_python_scripts', 'test_python_scripts', 'aivc', 'bat_scripts']

# both the escaped form used in normal strings and the single form used in raw strings
PATTERNS = [
    'D:\\\\data\\\\AIVC',   # "D:\\data\\AIVC"
    'D:\\data\\AIVC',       # r"D:\data\AIVC"
    'D:/data/AIVC',         # "D:/data/AIVC"
    '/mnt/d/data/AIVC',
]


def iter_files(root):
    for d in TARGET_DIRS:
        base = os.path.join(root, d)
        if not os.path.isdir(base):
            continue
        for r, _, fs in os.walk(base):
            for f in fs:
                if f.endswith('.py'):
                    yield os.path.join(r, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', required=True, help='absolute path to your AIVC checkout/data root')
    ap.add_argument('--dry-run', action='store_true', help='report changes without writing')
    args = ap.parse_args()

    target = args.root.rstrip('/\\')
    if not os.path.isabs(target):
        sys.exit('--root must be an absolute path')
    # Use forward slashes so non-raw Python strings on Windows (e.g. "C:\Users\...") never trigger unicodeescape errors
    target_safe = target.replace('\\', '/')

    changed = 0
    total_hits = 0
    for path in iter_files(os.getcwd()):
        try:
            s = open(path, encoding='utf-8').read()
        except UnicodeDecodeError:
            continue
        new = s
        hits = 0
        for pat in PATTERNS:
            hits += new.count(pat)
            new = new.replace(pat, target_safe)
        if hits == 0:
            continue
        total_hits += hits
        changed += 1
        print(f'{hits:4d}  {os.path.relpath(path)}')
        if not args.dry_run:
            open(path + '.bak', 'w', encoding='utf-8').write(s)
            open(path, 'w', encoding='utf-8').write(new)

    verb = 'would change' if args.dry_run else 'changed'
    print(f'\n{verb} {changed} files, {total_hits} path occurrences -> {target}')
    if not args.dry_run and changed:
        print('Originals saved as *.bak next to each modified file.')


if __name__ == '__main__':
    main()