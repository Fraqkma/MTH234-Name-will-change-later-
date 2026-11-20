"""Render the report draft as print-ready HTML without extra dependencies."""

from html import escape
from pathlib import Path


def main() -> None:
    lines = Path("report/draft.md").read_text(encoding="utf-8").splitlines()
    body: list[str] = []
    in_table = False
    for line in lines:
        if line.startswith("|"):
            if set(line.replace("|", "").replace("-", "").replace(":", "").strip()) == set():
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
            cells = (escape(cell.strip()) for cell in line.strip("|").split("|"))
            body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("# "):
            body.append(f"<h1>{escape(line[2:])}</h1>")
        elif line.startswith("## "):
            body.append(f"<h2>{escape(line[3:])}</h2>")
        elif line.strip():
            body.append(f"<p>{escape(line)}</p>")
    if in_table:
        body.append("</table>")
    style = ("@page{size:A4;margin:18mm}body{font:11pt/1.45 Georgia,serif;color:#172334;"
             "max-width:800px;margin:auto}h1,h2{font-family:Arial,sans-serif}h1{font-size:22pt}"
             "h2{font-size:15pt;margin-top:1.4em}table{border-collapse:collapse;width:100%;"
             "font:8pt Arial,sans-serif;margin:1em 0}td{padding:5px;border-bottom:1px solid #bbb}"
             "tr:first-child{background:#e9eef3;font-weight:bold}tr{break-inside:avoid}"
             "p{orphans:3;widows:3}")
    html = (f'<!doctype html><html lang="en"><meta charset="utf-8"><title>Report</title>'
            f'<style>{style}</style><body>' + "\n".join(body) + "</body></html>")
    Path("report/print.html").write_text(html, encoding="utf-8")


if __name__ == "__main__":
    main()
