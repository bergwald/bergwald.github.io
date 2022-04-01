"""
Quick and dirty script to convert a Jupyter notebook into a post
"""

import json

SOURCE = "/home/thomas/Git/ai/neuralnetworks.ipynb"
TARGET = "/_posts/article.md"

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
