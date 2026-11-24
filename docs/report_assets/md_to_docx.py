"""Convert docs/technical_report.md to a .docx (the challenge's report format).

Handles the subset of Markdown the report uses: headings, paragraphs, bullet and numbered lists (two levels),
pipe tables, images, horizontal rules, and inline **bold**, *italic* and `code`.

    python docs/report_assets/md_to_docx.py docs/technical_report.md docs/technical_report.docx
Needs: pip install python-docx
"""
import re
import sys
from pathlib import Path

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

INLINE = re.compile(r"(\*\*[^*]+\*\*|`[^`]+`|\*[^*\s][^*]*\*)")


def add_inline(par, text, bold=False):
    for part in INLINE.split(text):
        if not part:
            continue
        if part.startswith("**") and part.endswith("**"):
            run = par.add_run(part[2:-2]); run.bold = True
        elif part.startswith("`") and part.endswith("`"):
            run = par.add_run(part[1:-1]); run.font.name = "Consolas"; run.font.size = Pt(9)
            run._element.rPr.rFonts.set(qn("w:eastAsia"), "Consolas")
            run.bold = bold or None
        elif part.startswith("*") and part.endswith("*") and len(part) > 2:
            run = par.add_run(part[1:-1]); run.italic = True; run.bold = bold or None
        else:
            run = par.add_run(part); run.bold = bold or None


def shade(cell, hex_fill):
    tc_pr = cell._element.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear"); shd.set(qn("w:color"), "auto"); shd.set(qn("w:fill"), hex_fill)
    tc_pr.append(shd)


def add_table(doc, rows):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    header, body = cells[0], [r for r in cells[2:]]
    has_header = any(header)
    rows_all = ([header] if has_header else []) + body
    table = doc.add_table(rows=len(rows_all), cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for i, row in enumerate(rows_all):
        for j in range(len(header)):
            text = row[j] if j < len(row) else ""
            cell = table.cell(i, j)
            par = cell.paragraphs[0]
            par.paragraph_format.space_after = Pt(0)
            add_inline(par, text, bold=(i == 0 and has_header))
            for run in par.runs:
                run.font.size = Pt(8.5)
            if i == 0 and has_header:
                shade(cell, "DCE6F0")
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def convert(md_path: Path, out_path: Path):
    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    for side in ("left_margin", "right_margin"):
        setattr(sec, side, Cm(2.0))
    sec.top_margin = sec.bottom_margin = Cm(1.8)
    normal = doc.styles["Normal"]
    normal.font.name = "Calibri"; normal.font.size = Pt(10.5)
    normal.paragraph_format.space_after = Pt(4)
    for name, size in (("Heading 1", 15), ("Heading 2", 13), ("Heading 3", 11.5)):
        st = doc.styles[name]; st.font.size = Pt(size); st.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)

    lines = md_path.read_text(encoding="utf-8").splitlines()
    i, first_h1 = 0, True
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            i += 1; continue
        if line.startswith("|"):
            block = []
            while i < len(lines) and lines[i].startswith("|"):
                block.append(lines[i]); i += 1
            add_table(doc, block); continue
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", line.strip())
        if m:
            img = (md_path.parent / m.group(2)).resolve()
            doc.add_picture(str(img), width=Cm(16.5))
            doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap = doc.add_paragraph(); cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            run = cap.add_run(f"Figure: {m.group(1)}"); run.italic = True; run.font.size = Pt(9)
            i += 1; continue
        if line.strip() == "---":
            i += 1; continue
        h = re.match(r"(#{1,3}) (.*)", line)
        if h:
            level = len(h.group(1))
            if level == 1 and first_h1:
                title = doc.add_paragraph(); title.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = title.add_run(h.group(2)); run.bold = True; run.font.size = Pt(18)
                run.font.color.rgb = RGBColor(0x1F, 0x3A, 0x5F)
                first_h1 = False
            else:
                doc.add_heading(h.group(2).replace("`", ""), level=max(1, level - 1))
            i += 1; continue
        b = re.match(r"( *)(- |\d+\. )(.*)", line)
        if b:
            nested = len(b.group(1)) >= 2
            numbered = b.group(2)[0].isdigit()
            if numbered:  # literal numbers: Word's auto-numbering would continue across separate lists
                par = doc.add_paragraph()
                fmt = par.paragraph_format
                fmt.left_indent = Cm(1.27 if nested else 0.63); fmt.first_line_indent = Cm(-0.63)
                fmt.space_after = Pt(1)
                fmt.tab_stops.add_tab_stop(fmt.left_indent)
                par.add_run(b.group(2).strip() + "	")
            else:
                par = doc.add_paragraph(style="List Bullet" + (" 2" if nested else ""))
                par.paragraph_format.space_after = Pt(1)
            add_inline(par, b.group(3)); i += 1; continue
        par = doc.add_paragraph()
        if line.startswith("**") and i < 4:
            par.alignment = WD_ALIGN_PARAGRAPH.CENTER
        add_inline(par, line.strip()); i += 1
    doc.save(out_path)


if __name__ == "__main__":
    convert(Path(sys.argv[1]), Path(sys.argv[2]))
    print("wrote", sys.argv[2])
