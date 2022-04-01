"""
Quick and dirty script to convert a Jupyter notebook into a post
"""

import json

SOURCE = "/home/thomas/Git/ai/neuralnetworks.ipynb"
TARGET = "/_posts/article.md"

def format_mathjax(s):
    """Replace $ with \( and \)"""
    idx = 0
    open = False
    lenght = len(s)
    while idx < lenght:
        char = s[idx]
        next = s[idx+1] if idx+1 != lenght else ""
        if char == "$" and next != "$" and s[idx-1] != "$":
            if not open:
                s = s[:idx] + r"\\(" + s[idx+1:]
                open = True
            else:
                s = s[:idx] + r"\\)" + s[idx+1:]
                open = False
            idx += 1
            lenght += 2
        idx += 1
    return s

with open(SOURCE) as f:
    file = f.read()

notebook = json.loads(file)
cells = notebook["cells"]

with open(TARGET, "w") as f:
    for cell in cells:
        ctype = cell["cell_type"]
        match ctype:
            case "markdown":
                # Write text
                for text in cell["source"]:
                    text = format_mathjax(text)
                    f.write(text)
            case "code":
                # Write code
                f.write("```python\n")
                for code in cell["source"]:
                    f.write(code)
                f.write("\n```")
                # Write output
                if cell.get("outputs"):
                    for output in cell["outputs"]:
                        if output["output_type"] == "stream":
                            f.write('\n<div class="notice output">\n')
                            for out in output["text"]:
                                f.write(out)
                            f.write("</div>")
                        elif output["output_type"] == "execute_result":
                            f.write('\n<div class="notice output">\n')
                            for out in output.get("data").get("text/plain"):
                                f.write(out+"\n")
                            f.write("</div>")
        f.write("\n\n")
