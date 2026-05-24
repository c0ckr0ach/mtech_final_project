import json

with open("main.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

with open("results_v2.txt", "w", encoding="utf-8") as out_f:
    out_f.write(f"Number of cells: {len(nb['cells'])}\n")
    for idx, cell in enumerate(nb["cells"]):
        if cell["cell_type"] == "code":
            outputs = cell.get("outputs", [])
            if outputs:
                text_outputs = []
                for out in outputs:
                    if out.get("output_type") == "stream" and "text" in out:
                        val = out["text"]
                        if isinstance(val, list):
                            text_outputs.extend(val)
                        else:
                            text_outputs.append(str(val))
                    elif out.get("output_type") == "execute_result" and "data" in out:
                        if "text/plain" in out["data"]:
                            val = out["data"]["text/plain"]
                            if isinstance(val, list):
                                text_outputs.extend(val)
                            else:
                                text_outputs.append(str(val))
                    elif "text" in out:
                        val = out["text"]
                        if isinstance(val, list):
                            text_outputs.extend(val)
                        else:
                            text_outputs.append(str(val))
                
                combined_out = "".join(text_outputs)
                if "relevancy" in combined_out.lower() or "score" in combined_out.lower() or "eval" in combined_out.lower() or "delta" in combined_out.lower() or "mistral" in combined_out.lower():
                    out_f.write(f"--- Cell {idx} (Code) ---\n")
                    out_f.write("Source:\n")
                    out_f.write("".join(cell["source"][:10]) + "\n")
                    out_f.write("\nOutput snippet:\n")
                    out_f.write(combined_out[:3000] + "\n")
                    out_f.write("-" * 50 + "\n")
