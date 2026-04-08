from __future__ import annotations

import argparse
import re
from pathlib import Path

from pdfminer.high_level import extract_text


def normalize_line(line: str) -> str:
	return re.sub(r"\s+", " ", line).strip()


def is_heading(line: str) -> bool:
	if not line:
		return False

	if re.match(r"^(\d+(\.\d+)*)\s+[A-Za-z].*", line):
		return True

	alpha_chars = [char for char in line if char.isalpha()]
	if not alpha_chars:
		return False

	uppercase_ratio = sum(1 for char in alpha_chars if char.isupper()) / len(alpha_chars)
	return len(line) <= 90 and uppercase_ratio > 0.8


def markdown_from_text(text: str, source_name: str) -> str:
	pages = [page for page in text.split("\x0c") if page.strip()]
	output_lines: list[str] = [f"# {source_name}", ""]

	for page_index, page_text in enumerate(pages, start=1):
		output_lines.append(f"## Page {page_index}")
		output_lines.append("")

		raw_lines = page_text.splitlines()
		paragraph_parts: list[str] = []

		def flush_paragraph() -> None:
			if paragraph_parts:
				paragraph = " ".join(paragraph_parts).strip()
				if paragraph:
					output_lines.append(paragraph)
					output_lines.append("")
				paragraph_parts.clear()

		for raw_line in raw_lines:
			line = normalize_line(raw_line)

			if not line:
				flush_paragraph()
				continue

			bullet_match = re.match(r"^[•\-\*]\s+(.*)", line)
			if bullet_match:
				flush_paragraph()
				output_lines.append(f"- {bullet_match.group(1).strip()}")
				continue

			if is_heading(line):
				flush_paragraph()
				output_lines.append(f"### {line}")
				output_lines.append("")
				continue

			paragraph_parts.append(line)

		flush_paragraph()

	while output_lines and not output_lines[-1].strip():
		output_lines.pop()

	return "\n".join(output_lines) + "\n"


def convert_pdf_file(input_pdf: Path, output_md: Path) -> None:
	text = extract_text(str(input_pdf))
	markdown = markdown_from_text(text=text, source_name=input_pdf.stem)
	output_md.parent.mkdir(parents=True, exist_ok=True)
	output_md.write_text(markdown, encoding="utf-8")


def convert_path(input_path: Path, output_path: Path | None = None) -> list[Path]:
	written_files: list[Path] = []

	if input_path.is_file():
		if input_path.suffix.lower() != ".pdf":
			raise ValueError(f"Input file must be a PDF: {input_path}")

		target = output_path or input_path.with_suffix(".md")
		if target.suffix.lower() != ".md":
			target = target.with_suffix(".md")

		convert_pdf_file(input_path, target)
		written_files.append(target)
		return written_files

	if not input_path.is_dir():
		raise ValueError(f"Input path does not exist: {input_path}")

	pdf_files = sorted(input_path.rglob("*.pdf"))
	if not pdf_files:
		raise ValueError(f"No PDF files found in directory: {input_path}")

	output_root = output_path or input_path
	for pdf_file in pdf_files:
		relative_path = pdf_file.relative_to(input_path)
		target = (output_root / relative_path).with_suffix(".md")
		convert_pdf_file(pdf_file, target)
		written_files.append(target)

	return written_files


def build_arg_parser() -> argparse.ArgumentParser:
	parser = argparse.ArgumentParser(description="Convert PDF files to Markdown.")
	parser.add_argument("input", help="Path to a PDF file or a directory of PDF files.")
	parser.add_argument(
		"-o",
		"--output",
		help="Output .md path for file input, or output directory for folder input.",
	)
	return parser


def main() -> None:
	parser = build_arg_parser()
	args = parser.parse_args()

	input_path = Path(args.input).expanduser().resolve()
	output_path = Path(args.output).expanduser().resolve() if args.output else None

	written_files = convert_path(input_path=input_path, output_path=output_path)
	print(f"Converted {len(written_files)} PDF(s):")
	for output_file in written_files:
		print(f"- {output_file}")


if __name__ == "__main__":
	main()
