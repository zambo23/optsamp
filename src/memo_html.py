"""
Render the memo's Markdown as a standalone HTML page
────────────────────────────────────────────────────
Handles only the Markdown subset that build_memo() emits, so no extra
dependency is needed:
  headings (#, ##, ###) · horizontal rule (---) · paragraphs
  bullet lists (nested by 2 spaces) · numbered lists · pipe tables
  images on their own line (embedded as base64, so the file is portable)
  inline **bold**, *italic*, `code`
"""

import base64
import html
import mimetypes
import re
from pathlib import Path

CSS = """
body  { font: 15px/1.55 -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
        color: #222; max-width: 980px; margin: 40px auto; padding: 0 24px; }
h1    { font-size: 1.6em; border-bottom: 2px solid #378ADD; padding-bottom: .3em; }
h2    { font-size: 1.25em; margin-top: 2em; color: #185FA5; }
h3    { font-size: 1.05em; margin-top: 1.5em; }
hr    { border: 0; border-top: 1px solid #ddd; margin: 1.5em 0; }
code  { font: .9em Menlo, Consolas, monospace; background: #f3f3f1;
        padding: .1em .3em; border-radius: 3px; }
table { border-collapse: collapse; margin: 1em 0; font-size: .9em; }
th, td{ border: 1px solid #ddd; padding: .35em .7em; text-align: left; }
th    { background: #f3f6fa; }
tr:nth-child(even) td { background: #fafafa; }
img   { max-width: 100%; margin: .8em 0; }
li    { margin: .25em 0; }
.meta { color: #555; }
"""


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<em>\1</em>", text)
    return text


def embed_image(alt: str, src: str, img_dir: Path) -> str:
    path = img_dir / src
    if not path.exists():
        return f'<img alt="{html.escape(alt)}" src="{html.escape(src)}">'
    mime = mimetypes.guess_type(path.name)[0] or "image/png"
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img alt="{html.escape(alt)}" src="data:{mime};base64,{data}">'


def table(rows: list) -> str:
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    head, body = cells[0], cells[2:]          # cells[1] is the --- separator
    out = ["<table>", "<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in head)
           + "</tr></thead>", "<tbody>"]
    out += ["<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body]
    out += ["</tbody>", "</table>"]
    return "\n".join(out)


def markdown_to_html(md: str, title: str, img_dir: Path) -> str:
    lines = md.splitlines()
    out, para = [], []
    stack = []                                 # open lists: (indent, "ul"/"ol")

    def flush_para():
        if para:
            # several "**Key:** value" lines in a row = the memo's header block
            is_meta = len(para) > 1 and all(re.match(r"\*\*[^*]+:\*\*", p) for p in para)
            cls = ' class="meta"' if is_meta else ""
            out.append(f"<p{cls}>" + "<br>\n".join(inline(p) for p in para) + "</p>")
            para.clear()

    def close_lists(to_indent: int = -1):
        while stack and stack[-1][0] > to_indent:
            out.append(f"</li></{stack.pop()[1]}>")

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        item = re.match(r"^(\s*)(?:([-*])|(\d+)\.)\s+(.*)$", line)

        if not stripped:
            flush_para()
            # a blank line ends a list unless the next line continues it
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not re.match(r"^\s*(?:[-*]|\d+\.)\s+", nxt):
                close_lists()
        elif item and stripped != "---":
            flush_para()
            indent, kind = len(item.group(1)), ("ul" if item.group(2) else "ol")
            if stack and indent == stack[-1][0] and kind == stack[-1][1]:
                out.append("</li>")
            elif stack and indent <= stack[-1][0]:
                close_lists(indent)
                if stack and indent == stack[-1][0]:
                    out.append("</li>")
                else:
                    out.append(f"<{kind}>")
                    stack.append((indent, kind))
            else:
                out.append(f"<{kind}>")
                stack.append((indent, kind))
            out.append("<li>" + inline(item.group(4)))
        elif stripped.startswith("#"):
            flush_para(); close_lists()
            level = len(stripped) - len(stripped.lstrip("#"))
            out.append(f"<h{level}>{inline(stripped[level:].strip())}</h{level}>")
        elif stripped == "---":
            flush_para(); close_lists()
            out.append("<hr>")
        elif stripped.startswith("|"):
            flush_para(); close_lists()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i]); i += 1
            out.append(table(rows))
            continue
        elif (img := re.fullmatch(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)):
            flush_para(); close_lists()
            out.append(embed_image(img.group(1), img.group(2), img_dir))
        else:
            if stack:                          # continuation of a list item
                out.append(" " + inline(stripped))
            else:
                para.append(stripped)
        i += 1

    flush_para(); close_lists()
    body = "\n".join(out)
    return (f'<!DOCTYPE html>\n<html lang="en">\n<head>\n<meta charset="utf-8">\n'
            f"<title>{html.escape(title)}</title>\n<style>{CSS}</style>\n</head>\n"
            f"<body>\n{body}\n</body>\n</html>\n")
