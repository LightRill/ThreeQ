"""Enhance the uploaded midterm report with current ThreeQ experiment results.

The script preserves the original DOCX cover, formulas, and existing figures,
then inserts a consolidated experimental supplement before Section 3 and
updates the future-work and difficulty sections.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement
from docx.shared import Inches
from docx.text.paragraph import Paragraph


ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "paper_assets" / "figures"


def paragraph_after(paragraph: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def paragraph_before(paragraph: Paragraph, text: str = "", style: str | None = None) -> Paragraph:
    new_p = OxmlElement("w:p")
    paragraph._p.addprevious(new_p)
    new_para = Paragraph(new_p, paragraph._parent)
    if style:
        new_para.style = style
    if text:
        new_para.add_run(text)
    return new_para


def insert_table_before(target: Paragraph, rows: list[list[str]], style: str = "Normal Table") -> None:
    body = target._parent
    table = body.add_table(rows=len(rows), cols=len(rows[0]), width=Inches(6.2))
    table.style = style
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = True
    for i, row in enumerate(rows):
        for j, value in enumerate(row):
            cell = table.cell(i, j)
            cell.text = value
            for paragraph in cell.paragraphs:
                paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
            if i == 0:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.bold = True
    target._p.addprevious(table._tbl)


def insert_picture_before(target: Paragraph, image: Path, caption: str, width: float = 5.85) -> None:
    pic_para = paragraph_before(target)
    pic_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = pic_para.add_run()
    run.add_picture(str(image), width=Inches(width))
    cap_style = "图名中文" if "图名中文" in [style.name for style in target.part.document.styles] else "Normal"
    cap_para = paragraph_before(target, caption, cap_style)
    cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER


def find_paragraph(doc: Document, prefix: str) -> Paragraph:
    for paragraph in doc.paragraphs:
        if paragraph.text.strip().startswith(prefix):
            return paragraph
    raise ValueError(f"Cannot find paragraph starting with: {prefix}")


def replace_paragraph(paragraph: Paragraph, text: str) -> None:
    paragraph.clear()
    paragraph.add_run(text)


def insert_supplement(doc: Document) -> None:
    target = find_paragraph(doc, "3  后期拟完成")

    paragraph_before(target, "2.2.7 基于现有实验结果的综合补充", "Heading 3")
    paragraph_before(
        target,
        "在原有理论推导与收敛性实验基础上，本阶段进一步整理了 ThreeQ、EPThreeQ 与 DThreeQ 的可比较实验结果。考虑到当前仍有若干实验路线尚不充分，报告正文重点放在理论闭环较清楚、实验效果相对较好的部分：inference 收敛性验证、EPThreeQ 的稳定提升，以及 DThreeQ 在 CE-style 输出监督和 full-data restore-best 设置下的性能改进。",
        "Normal",
    )

    insert_picture_before(
        target,
        FIG / "fig03_mnist_current_performance.png",
        "图2.9  MNIST 当前性能总览：legacy EPThreeQ 与 DThreeQ CE/full-data 变体明显优于小预算公共 ThreeQ screen。",
    )
    insert_table_before(
        target,
        [
            ["模型/实验", "设置", "主要结果", "结论"],
            ["Base3Q legacy", "MNIST 5k/1k, 8 epoch", "60.2% best valid accuracy", "原始 ThreeQ 可学习但收敛较慢"],
            ["EPBase3Q legacy", "MNIST 5k/1k, 8 epoch", "77.5% best valid accuracy", "EP 明显优于 direct"],
            ["EPBase3Q tune", "MNIST 10k/2k, 15 epoch", "86.5% best valid accuracy", "当前最稳的原始 ThreeQ 改进线"],
            ["DThreeQ EP-nudge", "MNIST 10k/2k, 30 epoch", "83.9% best accuracy", "DThreeQ 可学习，但仍低于 EPThreeQ tune"],
            ["DThreeQ CE-nudge", "MNIST 10k/2k, 30 epoch", "86.5% selected accuracy", "CE-style 输出监督是明确正向改动"],
            ["DThreeQ full-data restore-best", "MNIST 60k/10k, 30 epoch", "88.0% selected accuracy", "接近 90%，但 final 回退到约 83.8%"],
        ],
    )
    paragraph_before(
        target,
        "表2.2  关键性能实验结果汇总。该表区分了 legacy 复现实验、小预算 stress screen 和 DThreeQ full-data restore-best，因此不能把不同预算下的结果直接混为同一个 benchmark。",
        "Normal",
    ).alignment = WD_ALIGN_PARAGRAPH.CENTER

    paragraph_before(target, "2.2.8 DThreeQ 改进过程与实验现象", "Heading 3")
    paragraph_before(
        target,
        "DThreeQ 的实验结果表明，小预算公共框架下的失败不能直接否定该结构。随着 hidden size、训练预算、输出监督方式和 full-data 训练逐步增强，DThreeQ 从接近随机逐步提升到 10k/2k 上约 86.5% selected accuracy，并在 full-data restore-best 设置下达到约 88.0%。但是该结果仍存在明显后期崩塌，说明当前方法需要保存 best checkpoint，并进一步控制状态饱和、输入能量比例和学习率后期漂移。",
        "Normal",
    )
    insert_picture_before(
        target,
        FIG / "fig04_dthreeq_improvement_process.png",
        "图2.10  DThreeQ 从小预算失败到 full-data restore-best 的改进过程。",
    )
    insert_picture_before(
        target,
        FIG / "fig05_dthreeq_training_curves.png",
        "图2.11  DThreeQ 10k 与 full-data 训练曲线，显示 early-best 与后期崩塌现象。",
    )
    insert_table_before(
        target,
        [
            ["阶段", "变体", "数据预算", "best acc.", "final acc.", "解释"],
            ["small public screen", "dthreeq_ep_nudge_0p01", "3k/1k, 3 epochs", "13.32%", "13.32%", "小预算接近随机，不能代表原始设定"],
            ["legacy-scale focus", "dthreeq_ep_nudge0p1_lr1e3", "10k/2k, 12 epochs", "80.35%", "80.35%", "扩大规模后明显可学习"],
            ["longrun EP-nudge", "dthreeq_ep_nudge0p1_lr3e3", "10k/2k, 30 epochs", "83.94%", "83.54%", "当前稳定 baseline"],
            ["CE-style nudge", "dthreeq_ep_ce_nudge0p1_lr3e3", "10k/2k, 30 epochs", "86.84%", "86.55%", "输出监督改为 CE 后提升明显"],
            ["full-data restore-best", "dthreeq_ep_signed_nudge0p1_lr3e3_restorebest", "60k/10k, 30 epochs", "88.03%", "83.79%", "best checkpoint 较好，但 final 回退"],
        ],
    )
    paragraph_before(target, "表2.3  DThreeQ 改进过程分阶段摘要。", "Normal").alignment = WD_ALIGN_PARAGRAPH.CENTER

    paragraph_before(target, "2.2.9 输入能量与输出监督分析", "Heading 3")
    paragraph_before(
        target,
        "MNIST 实验还显示，784 维输入重构项会影响 10 维分类监督，但简单删除输入项并不能解决问题。输入 residual weight 设为 0、1/784 或 1/28 会降低最终 accuracy，说明输入项既可能压制监督，也在维持可用表示。相对而言，CE-style output nudge 是当前最明确的正向改动：在 10k/2k、30 epoch 设置下达到约 86.5% selected accuracy，并在 full-data restore-best 设置下接近 88.0%。",
        "Normal",
    )
    insert_picture_before(
        target,
        FIG / "fig08_input_energy_supervision.png",
        "图2.12  输入重构能量比例与分类 accuracy 的关系。",
    )


def update_future_work(doc: Document) -> None:
    schedule = find_paragraph(doc, "第十二～十四周")
    replace_paragraph(schedule, "第十二～十四周：完成对称结构理论分析，推广到更一般的对称局部预测结构，进行数值实验验证；")
    p = find_paragraph(doc, "具体计划如下")
    replace_paragraph(
        p,
        "具体计划调整如下：一是将 EPBase3Q 的 beta=1.0 与 weak_steps=5 扩展到更完整的 MNIST benchmark，并采用多 seed 与 best checkpoint 记录；二是在公共框架中逐项对齐 legacy hidden size、batch size、状态步数、epsilon、beta、alphas 和初始化，找出 small-budget 退化来源；三是继续推进 DThreeQ 的 CE/softmax 输出监督、boundary-clamped input、layer-normalized energy 与 conservative learning-rate schedule，目标是保住 full-data early-best 而不发生后期崩塌；四是保留 Dplus 机制诊断作为问题定位工具，但不把当前 residual-delta 结果作为主要性能结论，后续只在方向指标明显改善后再进入训练筛查。",
    )


def update_difficulties(doc: Document) -> None:
    p = find_paragraph(doc, "目前的主要困难已经不再是局部 inference")
    replace_paragraph(
        p,
        "目前的主要困难已经不再是局部 inference 稳定性的证明，而是“推断能够稳定收敛，但训练性能、表达能力和跨层 credit assignment 仍然不足”的矛盾。理论上，谱半径条件可以保证局部状态推断稳定；但在实现层面，hard clipping、状态饱和、输入重构能量占比过高、局部平均预测和 detach 路径都会削弱监督信号向浅层传播。",
    )
    q = paragraph_after(
        p,
        "具体而言，第一，EPThreeQ 虽然比 direct ThreeQ 稳定，但仍存在 early-best 后性能回退和训练成本较高的问题；第二，DThreeQ full-data restore-best 已接近 88.0% accuracy，但 final accuracy 会回退到约 83.8%，说明需要额外的状态稳定化与学习率调度；第三，Dplus 当前与 BP 更新方向仍不对齐，因此在中期报告中只作为机制问题和后续优化方向简要说明，不作为主要实验成果展开。",
        "Normal",
    )
    paragraph_after(
        q,
        "因此，后续论文应把“收敛性证明”“训练规则有效性”“高维任务性能”和“对称结构推广”分开论证，避免把局部稳定性误写成全局训练有效性。",
        "Normal",
    )


def enhance(input_path: Path, output_path: Path) -> None:
    doc = Document(input_path)
    insert_supplement(doc)
    update_future_work(doc)
    update_difficulties(doc)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    enhance(args.input, args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
