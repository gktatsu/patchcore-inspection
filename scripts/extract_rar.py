#!/usr/bin/env python3
"""Extract .rar archives using available system tools.

【使い方】(ステップバイステップ)
1) 前提条件：以下のいずれかのツールをインストールしておく → unrar, 7z, bsdtar
2) 単一のRARファイルを展開：
   python scripts/extract_rar.py /path/to/file.rar -o /path/to/output
3) ディレクトリ内の全RARファイルを再帰的に展開：
   python scripts/extract_rar.py /path/to/rar_dir -r -o /path/to/output
4) オプションフラグ：
   --keep-structure  : 元のディレクトリ構造を出力先で保持
   --overwrite       : 既存ファイルを上書き

【実行例】
   python scripts/extract_rar.py path/to/data.rar
   python scripts/extract_rar.py path/to/rar_dir --recursive -o ./extracted
"""

# Legacy usage comment:
# 1) Ensure at least one extractor is installed: unrar or 7z or bsdtar.
# 2) Extract one archive: python scripts/extract_rar.py /path/to/file.rar -o /path/to/output
# 3) Extract all archives in a directory: python scripts/extract_rar.py /path/to/rar_dir -r -o /path/to/output
# 4) Optional flags: --keep-structure, --overwrite

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract .rar archives")
    parser.add_argument(
        "input_path",
        type=Path,
        help="RAR file path or directory containing .rar files",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Base output directory (default: current directory)",
    )
    parser.add_argument(
        "-r",
        "--recursive",
        action="store_true",
        help="Recursively search for .rar files when input_path is a directory",
    )
    parser.add_argument(
        "--keep-structure",
        action="store_true",
        help="Keep input directory structure under output-dir",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files when extractor supports it",
    )
    return parser.parse_args()


def find_extractor() -> str:
    for command in ("unrar", "7z", "bsdtar"):
        if shutil.which(command):
            path = shutil.which(command)
            try:
                version_result = subprocess.run(
                    [command, "--version"] if command != "7z" else [command, "i"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    universal_newlines=True,
                )
                version_line = (version_result.stdout or version_result.stderr or "").splitlines()
                version_info = version_line[0] if version_line else "unknown version"
            except Exception:
                version_info = "unknown version"
            print(f"[extractor] Using '{command}' ({path}) — {version_info}", file=sys.stderr)
            if command == "7z":
                print(
                    "[extractor] WARNING: p7zip/7z has incomplete RAR5 support. "
                    "If you see 'Unsupported Method' errors, install 'unrar': sudo apt install unrar",
                    file=sys.stderr,
                )
            return command
    raise FileNotFoundError(
        "No supported extractor found. Install one of: unrar, 7z, bsdtar."
    )


def collect_archives(input_path: Path, recursive: bool) -> List[Path]:
    if input_path.is_file():
        if input_path.suffix.lower() != ".rar":
            raise ValueError(f"Input file is not a .rar archive: {input_path}")
        return [input_path]

    if not input_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    pattern = "**/*.rar" if recursive else "*.rar"
    archives = sorted(input_path.glob(pattern))
    if not archives:
        raise FileNotFoundError(f"No .rar files found in: {input_path}")
    return archives


def build_command(
    extractor: str,
    archive: Path,
    destination: Path,
    overwrite: bool,
) -> List[str]:
    if extractor == "unrar":
        overwrite_flag = "-o+" if overwrite else "-o-"
        return ["unrar", "x", overwrite_flag, str(archive), str(destination)]

    if extractor == "7z":
        overwrite_flag = "-y" if overwrite else "-aos"
        return ["7z", "x", overwrite_flag, f"-o{destination}", str(archive)]

    if extractor == "bsdtar":
        command = ["bsdtar", "-xf", str(archive), "-C", str(destination)]
        if overwrite:
            command.insert(1, "--unlink-first")
        return command

    raise ValueError(f"Unsupported extractor: {extractor}")


def resolve_destination(
    archive: Path,
    input_path: Path,
    output_dir: Path,
    keep_structure: bool,
) -> Path:
    if input_path.is_file():
        return output_dir / archive.stem

    if keep_structure:
        relative_parent = archive.parent.relative_to(input_path)
        return output_dir / relative_parent / archive.stem

    return output_dir / archive.stem


def log_archive_info(extractor: str, archive: Path) -> None:
    """List archive contents/metadata for diagnostics."""
    if extractor == "7z":
        cmd = ["7z", "l", "-slt", str(archive)]
    elif extractor == "unrar":
        cmd = ["unrar", "lt", str(archive)]
    else:
        return
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    # Show only Method lines to reveal which compression methods are used
    method_lines = [ln for ln in result.stdout.splitlines() if "Method" in ln]
    methods = set(method_lines)
    if methods:
        print(f"[archive info] Compression methods found in {archive.name}:", file=sys.stderr)
        for m in sorted(methods):
            print(f"  {m}", file=sys.stderr)


def extract_one(
    extractor: str,
    archive: Path,
    destination: Path,
    overwrite: bool,
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    command = build_command(extractor, archive, destination, overwrite)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    if result.returncode != 0:
        stderr = result.stderr.strip() or "(no stderr)"
        stdout = result.stdout.strip() or "(no stdout)"
        # Count and summarize "Unsupported Method" errors specifically
        unsupported = [ln for ln in result.stderr.splitlines() if "Unsupported Method" in ln]
        hint = ""
        if unsupported:
            hint = (
                f"\n[hint] {len(unsupported)} file(s) failed with 'Unsupported Method'. "
                "This means the extractor does not support the compression algorithm used.\n"
                "[hint] Fix: install 'unrar' (sudo apt install unrar) which fully supports RAR5."
            )
            log_archive_info(extractor, archive)
        raise RuntimeError(
            f"Extraction failed for {archive}\n"
            f"Command: {' '.join(command)}\n"
            f"stdout: {stdout}\n"
            f"stderr: {stderr}"
            f"{hint}"
        )


def process_archives(
    extractor: str,
    archives: Iterable[Path],
    input_path: Path,
    output_dir: Path,
    keep_structure: bool,
    overwrite: bool,
) -> None:
    for archive in archives:
        destination = resolve_destination(
            archive=archive,
            input_path=input_path,
            output_dir=output_dir,
            keep_structure=keep_structure,
        )
        print(f"Extracting: {archive} -> {destination}")
        extract_one(extractor, archive, destination, overwrite)


def main() -> int:
    args = parse_args()

    try:
        extractor = find_extractor()
        archives = collect_archives(args.input_path, args.recursive)
        process_archives(
            extractor=extractor,
            archives=archives,
            input_path=args.input_path,
            output_dir=args.output_dir,
            keep_structure=args.keep_structure,
            overwrite=args.overwrite,
        )
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
