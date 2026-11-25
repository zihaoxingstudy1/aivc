# -*- coding: utf-8 -*-
"""
md_to_html_exporter.py
Convert Markdown documents into publication-grade interactive HTML files featuring:
  - Sticky left Sidebar Table of Contents (auto-generated from h2/h3)
  - Responsive horizontal-scroll table wrappers (zero viewport overflow)
  - Click-to-zoom full-screen Image Lightbox for 300-DPI Nature-grade figures
  - MathJax 3 LaTeX equation rendering
"""

import os
import re
import html
import markdown as md_lib


CSS_STYLE = """
:root {
  --bg: #f4f7fb;
  --paper: #ffffff;
  --text: #1e293b;
  --muted: #64748b;
  --accent: #b91c1c;
  --accent2: #1d4ed8;
  --border: #e2e8f0;
  --code-bg: #f1f5f9;
}
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Noto Sans SC", "Microsoft YaHei", Helvetica, Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.78;
  margin: 0;
  display: flex;
  min-height: 100vh;
}
.sidebar {
  width: 270px;
  flex-shrink: 0;
  background: #0f172a;
  color: #e2e8f0;
  padding: 24px 18px;
  position: sticky;
  top: 0;
  height: 100vh;
  overflow-y: auto;
  border-right: 1px solid #1e293b;
}
.sidebar h4 {
  margin: 0 0 14px;
  font-size: 0.85rem;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: #94a3b8;
  border-bottom: 1px solid #334155;
  padding-bottom: 8px;
}
.sidebar ul { list-style: none; padding: 0; margin: 0; }
.sidebar li { margin: 6px 0; }
.sidebar li.toc-h3 { padding-left: 14px; font-size: 0.82rem; }
.sidebar a {
  color: #cbd5e1;
  text-decoration: none;
  font-size: 0.86rem;
  display: block;
  padding: 3px 6px;
  border-radius: 5px;
  transition: background 0.15s, color 0.15s;
}
.sidebar a:hover { background: #1e293b; color: #ffffff; }
.main-wrap {
  flex: 1;
  min-width: 0;
  padding: 28px 24px 72px;
}
.container {
  max-width: 980px;
  margin: 0 auto;
  background: var(--paper);
  padding: 44px 52px;
  border-radius: 12px;
  box-shadow: 0 6px 28px rgba(15, 23, 42, 0.07);
  border: 1px solid var(--border);
}
h1 { font-size: 1.9rem; line-height: 1.32; color: #0f172a; border-bottom: 2.5px solid var(--accent); padding-bottom: 12px; margin-top: 0; }
h2 { font-size: 1.38rem; color: #0f172a; margin-top: 2.3rem; border-bottom: 1px solid var(--border); padding-bottom: 6px; scroll-margin-top: 16px; }
h3 { font-size: 1.12rem; color: #1e3a8a; margin-top: 1.5rem; scroll-margin-top: 16px; }
p { margin: 0.9rem 0; }
blockquote {
  margin: 1.1rem 0;
  padding: 12px 18px;
  background: #fef2f2;
  border-left: 4px solid var(--accent);
  color: #7f1d1d;
  border-radius: 0 8px 8px 0;
}
.table-wrap {
  width: 100%;
  overflow-x: auto;
  margin: 1.2rem 0;
  border-radius: 6px;
  border: 1px solid var(--border);
}
table {
  width: 100%;
  border-collapse: collapse;
  font-size: 0.88rem;
}
th, td {
  border-bottom: 1px solid var(--border);
  border-right: 1px solid var(--border);
  padding: 8px 11px;
  text-align: left;
  white-space: nowrap;
}
th:last-child, td:last-child { border-right: none; }
th { background: #f1f5f9; font-weight: 600; color: #0f172a; }
tr:nth-child(even) td { background: #fafcff; }
code {
  font-family: Consolas, "Courier New", monospace;
  background: var(--code-bg);
  padding: 2px 5px;
  border-radius: 4px;
  font-size: 0.87em;
}
pre {
  background: #0f172a;
  color: #e2e8f0;
  padding: 14px 18px;
  border-radius: 8px;
  overflow-x: auto;
  line-height: 1.5;
}
pre code { background: transparent; color: inherit; padding: 0; }
img {
  display: block;
  max-width: 100%;
  height: auto;
  margin: 18px auto 6px;
  border-radius: 6px;
  border: 1px solid var(--border);
  box-shadow: 0 3px 14px rgba(15, 23, 42, 0.08);
  background: #ffffff;
  cursor: zoom-in;
  transition: transform 0.15s;
}
img:hover { transform: scale(1.005); }
p em { color: var(--muted); font-size: 0.88rem; display: block; text-align: center; margin-bottom: 14px; }
hr { border: none; border-top: 1px solid var(--border); margin: 2rem 0; }
a { color: var(--accent2); text-decoration: none; }
a:hover { text-decoration: underline; }
#lightbox {
  display: none;
  position: fixed;
  z-index: 9999;
  inset: 0;
  background: rgba(15, 23, 42, 0.90);
  align-items: center;
  justify-content: center;
  padding: 24px;
  cursor: zoom-out;
}
#lightbox img {
  max-width: 96vw;
  max-height: 94vh;
  margin: 0;
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.5);
  cursor: zoom-out;
}
@media (max-width: 900px) {
  body { flex-direction: column; }
  .sidebar { width: 100%; height: auto; position: static; }
  .container { padding: 24px 18px; }
}
"""

JS_SCRIPT = """
<script>
document.addEventListener("DOMContentLoaded", () => {
  const tocList = document.getElementById("toc-list");
  const headings = document.querySelectorAll("main h2, main h3");
  headings.forEach((h, idx) => {
    const id = "sec-" + idx;
    h.id = id;
    const li = document.createElement("li");
    li.className = h.tagName === "H3" ? "toc-h3" : "toc-h2";
    const a = document.createElement("a");
    a.href = "#" + id;
    a.textContent = h.textContent;
    li.appendChild(a);
    tocList.appendChild(li);
  });
  const lb = document.getElementById("lightbox");
  const lbImg = document.getElementById("lightbox-img");
  document.querySelectorAll("main img").forEach(img => {
    img.addEventListener("click", () => {
      lbImg.src = img.src;
      lb.style.display = "flex";
    });
  });
  lb.addEventListener("click", () => { lb.style.display = "none"; });
});
</script>
"""


def convert_markdown_file_to_html(md_path: str, html_path: str, page_title: str = "") -> str:
    md_path = str(md_path)
    html_path = str(html_path)
    with open(md_path, "r", encoding="utf-8") as f:
        raw_md = f.read()

    if not page_title:
        m_title = re.search(r"^#\s+(.+)$", raw_md, flags=re.MULTILINE)
        page_title = m_title.group(1).strip() if m_title else os.path.basename(md_path)

    math_blocks = []
    def _save_math(match):
        math_blocks.append(match.group(0))
        return f"@@MATHBLOCK{len(math_blocks)-1}@@"

    protected_md = re.sub(r"\$\$[\s\S]+?\$\$", _save_math, raw_md)
    protected_md = re.sub(r"(?<!\$)\$(?!\$)(.+?)(?<!\$)\$(?!\$)", _save_math, protected_md)

    body_html = md_lib.markdown(
        protected_md,
        extensions=["tables", "fenced_code", "sane_lists"],
    )
    body_html = body_html.replace("<table>", '<div class="table-wrap"><table>').replace("</table>", "</table></div>")

    for i, mb in enumerate(math_blocks):
        body_html = body_html.replace(f"@@MATHBLOCK{i}@@", html.escape(mb))

    full_html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>{html.escape(page_title)}</title>
  <style>{CSS_STYLE}</style>
  <script>
    window.MathJax = {{
      tex: {{ inlineMath: [['$', '$']], displayMath: [['$$', '$$']] }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script defer src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
</head>
<body>
  <aside class="sidebar">
    <h4>Contents</h4>
    <ul id="toc-list"></ul>
  </aside>
  <div class="main-wrap">
    <main class="container">
      {body_html}
    </main>
  </div>
  <div id="lightbox"><img id="lightbox-img" src="" alt="Zoomed Figure" /></div>
  {JS_SCRIPT}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Converted {md_path} -> {html_path} ({os.path.getsize(html_path):,} bytes)")
    return html_path


if __name__ == "__main__":
    docs_dir = r"D:\data\AIVC\docs"
    for base in ["AIVC_Complete_Tutorial_CN", "AIVC_TopJournal_Manuscript_Draft"]:
        convert_markdown_file_to_html(
            os.path.join(docs_dir, f"{base}.md"),
            os.path.join(docs_dir, f"{base}.html"),
        )
