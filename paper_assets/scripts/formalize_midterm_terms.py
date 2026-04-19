"""Formalize terminology and shorten the difficulties section in a midterm DOCX."""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.text.paragraph import Paragraph


REPLACEMENTS = [
    ("EPBase3Q", "EP预测性树突网络"),
    ("Base3Q", "基础预测性树突网络"),
    ("EPThreeQ", "EP预测性树突网络"),
    ("DThreeQ", "动态预测性树突网络"),
    ("ThreeQ", "预测性树突网络"),
    ("direct 预测性树突网络", "direct 训练的预测性树突网络"),
    ("小预算公共原始网络", "小预算公共预测性树突网络"),
    ("Dynamic路线", "动态预测性树突网络路线"),
    ("公共原始网络", "公共预测性树突网络基线"),
    ("EP预测性树突网络 与 动态预测性树突网络", "EP预测性树突网络与动态预测性树突网络"),
    ("EP预测性树突网络 的", "EP预测性树突网络的"),
    ("动态预测性树突网络 的", "动态预测性树突网络的"),
    ("动态预测性树突网络 从", "动态预测性树突网络从"),
    ("原始 预测性树突网络 可学习", "原始预测性树突网络可学习"),
    ("原始 预测性树突网络", "原始预测性树突网络"),
    ("原始预测性树突网络 改进线", "预测性树突网络改进路线"),
    ("EP 明显优于 direct", "EP训练明显优于直接训练"),
]


def apply_replacements(text: str) -> str:
    for old, new in REPLACEMENTS:
        text = text.replace(old, new)
    return text


def replace_paragraph_runs(paragraph: Paragraph) -> None:
    # Preserve run-level formatting when possible.
    has_drawing = any(run._element.xpath(".//w:drawing") or run._element.xpath(".//w:pict") for run in paragraph.runs)
    if not has_drawing:
        text = paragraph.text
        replaced = apply_replacements(text)
        if replaced != text:
            paragraph.clear()
            paragraph.add_run(replaced)
        return
    for run in paragraph.runs:
        if run._element.xpath(".//w:drawing") or run._element.xpath(".//w:pict"):
            continue
        run.text = apply_replacements(run.text)


def paragraph_after(paragraph: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def remove_paragraph(paragraph: Paragraph) -> None:
    element = paragraph._element
    element.getparent().remove(element)


def find_heading(doc: Document, prefix: str) -> int:
    for i, paragraph in enumerate(doc.paragraphs):
        if paragraph.text.strip().startswith(prefix):
            return i
    raise ValueError(f"Cannot find heading: {prefix}")


def shorten_difficulties(doc: Document) -> None:
    start = find_heading(doc, "4  存在的问题")
    end = find_heading(doc, "5  如期完成")
    heading = doc.paragraphs[start]
    old_paragraphs = list(doc.paragraphs[start + 1 : end])
    for paragraph in old_paragraphs:
        remove_paragraph(paragraph)

    first = paragraph_after(
        heading,
        "目前研究的主要挑战集中在训练效率与性能进一步提升上。已有理论和实验已经较好地验证了预测性树突网络在谱半径条件下的局部推断收敛性；EP预测性树突网络和动态预测性树突网络的部分设置也已经在 MNIST 上取得较稳定结果。后续需要继续改进的是受扰相信号强度、输入能量归一化以及高维任务上的训练稳定性。",
        "Normal",
    )
    paragraph_after(
        first,
        "总体来看，这些问题属于后续优化和扩展阶段需要解决的技术问题，并不影响目前已完成的理论分析、收敛性验证和主要实验结论。下一阶段将优先围绕表现较好的 EP 训练路线和 CE-style 输出监督继续推进，同时保留机制诊断作为辅助分析工具。",
        "Normal",
    )


def formalize_docx(input_path: Path, output_path: Path) -> None:
    doc = Document(input_path)
    shorten_difficulties(doc)

    for paragraph in doc.paragraphs:
        replace_paragraph_runs(paragraph)
    for table in doc.tables:
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    replace_paragraph_runs(paragraph)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    formalize_docx(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
