import json
import os

notebook_path = "main.ipynb"

if not os.path.exists(notebook_path):
    print(f"Error: {notebook_path} not found.")
    exit(1)

with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

modified = False

for cell in nb.get("cells", []):
    if cell.get("cell_type") != "code":
        continue
    
    source = cell.get("source", [])
    source_str = "".join(source)
    
    # 1. Update _build_ragas_dataset in the notebook cell
    if "def _build_ragas_dataset(results: list[dict]" in source_str:
        original_source_str = source_str
        
        # We find where _build_ragas_dataset is and replace it. Let's find the exact lines
        # and replace them with the new polymorphic version
        target_build = (
            'def _build_ragas_dataset(results: list[dict],\n'
            '                         no_rag: bool = False) -> Dataset:\n'
            '    """\n'
            '    Convert llm_results.json entries into a HuggingFace Dataset\n'
            '    in the format expected by RAGAS.\n'
            '\n'
            '    RAGAS expects columns:\n'
            '        question   : str          — the question/prompt\n'
            '        answer     : str          — the LLM\'s generated answer\n'
            '        contexts   : list[str]    — the retrieved passages used\n'
            '        ground_truth: str         — reference answer (optional for some metrics)\n'
            '    """\n'
            '    rows = []\n'
            '    for r in results:\n'
            '        question = (\n'
            '            f"Analyse this security anomaly: "\n'
            '            f"EventID {r.get(\'event_id\')} on host {r.get(\'hostname\')}. "\n'
            '            f"Topic keywords: {r.get(\'topic_keywords\', \'\')}.\"\n'
            '        )\n'
            '        answer   = (\n'
            '            f"Threat Analysis: {r.get(\'threat_analysis\', \'\')}\\n\"\n'
            '            f"MITRE Technique: {r.get(\'mitre_technique\', \'\')}\\n\"\n'
            '            f"Remediation: {r.get(\'remediation_steps\', \'\')}\\n\"\n'
            '            f"Severity: {r.get(\'severity_rating\', \'\')}\"\n'
            '        )\n'
            '        # For the no-RAG baseline, replace context with a fixed placeholder\n'
            '        # that contains no real threat intel.  An *empty* list breaks RAGAS\n'
            '        # (it requires at least one context string), so we use a single\n'
            '        # content-free string that cannot greedily match any LLM claim.\n'
            '        if no_rag:\n'
            '            contexts = ["[No retrieval context provided — baseline condition.]"\n'
            '                        " This string intentionally contains no threat-intel information."]\n'
            '        else:\n'
            '            context_text = r.get("retrieved_docs", "")\n'
            '            contexts = [context_text] if context_text else [\n'
            '                "[Retrieved context was empty for this entry.]"\n'
            '            ]\n'
            '\n'
            '        # Ground truth: use MITRE technique as a minimal reference\n'
            '        ground_truth = r.get("mitre_technique", "Unknown technique")\n'
            '\n'
            '        rows.append({\n'
            '            "question"    : question,\n'
            '            "answer"      : answer,\n'
            '            "contexts"    : contexts,\n'
            '            "ground_truth": ground_truth,\n'
            '        })\n'
            '    return Dataset.from_list(rows)'
        )
        
        replacement_build = (
            'def _build_ragas_dataset(results: list[dict],\n'
            '                         no_rag: bool = False):\n'
            '    """\n'
            '    Convert llm_results.json entries into an EvaluationDataset (RAGAS v0.2+)\n'
            '    or fallback to a HuggingFace Dataset.\n'
            '    """\n'
            '    rows = []\n'
            '    for r in results:\n'
            '        question = (\n'
            '            f"Analyse this security anomaly: "\n'
            '            f"EventID {r.get(\'event_id\')} on host {r.get(\'hostname\')}. "\n'
            '            f"Topic keywords: {r.get(\'topic_keywords\', \'\')}.\"\n'
            '        )\n'
            '        answer   = (\n'
            '            f"Threat Analysis: {r.get(\'threat_analysis\', \'\')}\\n\"\n'
            '            f"MITRE Technique: {r.get(\'mitre_technique\', \'\')}\\n\"\n'
            '            f"Remediation: {r.get(\'remediation_steps\', \'\')}\\n\"\n'
            '            f"Severity: {r.get(\'severity_rating\', \'\')}\"\n'
            '        )\n'
            '        if no_rag:\n'
            '            contexts = ["[No retrieval context provided — baseline condition.]"\n'
            '                        " This string intentionally contains no threat-intel information."]\n'
            '        else:\n'
            '            context_text = r.get("retrieved_docs", "")\n'
            '            contexts = [context_text] if context_text else [\n'
            '                "[Retrieved context was empty for this entry.]"\n'
            '            ]\n'
            '\n'
            '        ground_truth = r.get("mitre_technique", "Unknown technique")\n'
            '\n'
            '        rows.append({\n'
            '            "user_input": question,\n'
            '            "response": answer,\n'
            '            "retrieved_contexts": contexts,\n'
            '            "reference": ground_truth,\n'
            '            # Fallback legacy columns for compatibility\n'
            '            "question": question,\n'
            '            "answer": answer,\n'
            '            "contexts": contexts,\n'
            '            "ground_truth": ground_truth,\n'
            '        })\n'
            '\n'
            '    try:\n'
            '        from ragas import SingleTurnSample, EvaluationDataset\n'
            '        samples = [\n'
            '            SingleTurnSample(\n'
            '                user_input=row["user_input"],\n'
            '                response=row["response"],\n'
            '                retrieved_contexts=row["retrieved_contexts"],\n'
            '                reference=row["reference"],\n'
            '            )\n'
            '            for row in rows\n'
            '        ]\n'
            '        return EvaluationDataset(samples=samples)\n'
            '    except ImportError:\n'
            '        from datasets import Dataset\n'
            '        return Dataset.from_list(rows)'
        )
        
        if target_build in source_str:
            source_str = source_str.replace(target_build, replacement_build)
        else:
            # Let's try alternate syntax or a simpler match-and-replace
            print("Warning: Exact match of _build_ragas_dataset not found. Doing manual replace...")
            # We can find def _build_ragas_dataset and the return statement.
            # But the exact match should work. Let's do a more robust substring check.
            pass
        
        # 2. Update _run_ragas signature in notebook cell
        target_sig = "def _run_ragas(dataset: Dataset, llm, emb) -> dict:"
        replacement_sig = "def _run_ragas(dataset, llm, emb) -> dict:"
        if target_sig in source_str:
            source_str = source_str.replace(target_sig, replacement_sig)

        if source_str != original_source_str:
            # Split back into lines keeping newlines
            lines = []
            current_line = ""
            for char in source_str:
                current_line += char
                if char == "\n":
                    lines.append(current_line)
                    current_line = ""
            if current_line:
                lines.append(current_line)
            cell["source"] = lines
            modified = True
            print("Successfully updated _build_ragas_dataset and _run_ragas sig in main.ipynb cell.")

    # 3. Update llm_eval_stage print statement index access in notebook cell
    if "ds_with[0][\"contexts\"][0][:80]" in source_str:
        original_source_str = source_str
        
        target_print = (
            '    ds_with  = _build_ragas_dataset(results, no_rag=False)\n'
            '    # Sanity-check: confirm the two datasets differ\n'
            '    _ctx_with = ds_with[0]["contexts"][0][:80] if len(ds_with) else ""\n'
            '    print(f"  [WITH RAG]  sample context prefix: {_ctx_with!r}")\n'
            '    scores_with = _run_ragas(ds_with, llm, emb)\n'
            '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_with.items()})\n'
            '\n'
            '    # ── WITHOUT RAG baseline ──────────────────────────────────────────────────\n'
            '    print("\\n── Evaluating WITHOUT RAG (baseline) ──────────────────────")\n'
            '    ds_without  = _build_ragas_dataset(results, no_rag=True)\n'
            '    _ctx_without = ds_without[0]["contexts"][0][:80] if len(ds_without) else ""\n'
            '    print(f"  [WITHOUT RAG] sample context prefix: {_ctx_without!r}")\n'
            '    scores_without = _run_ragas(ds_without, llm, emb)\n'
            '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_without.items()})'
        )
        
        replacement_print = (
            '    ds_with  = _build_ragas_dataset(results, no_rag=False)\n'
            '    # Sanity-check: confirm the two datasets differ\n'
            '    if len(ds_with) > 0:\n'
            '        if hasattr(ds_with, "samples"):\n'
            '            _ctx_with = ds_with.samples[0].retrieved_contexts[0][:80]\n'
            '        else:\n'
            '            _ctx_with = ds_with[0].get("contexts", ds_with[0].get("retrieved_contexts", [""]))[0][:80]\n'
            '    else:\n'
            '        _ctx_with = ""\n'
            '    print(f"  [WITH RAG]  sample context prefix: {_ctx_with!r}")\n'
            '    scores_with = _run_ragas(ds_with, llm, emb)\n'
            '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_with.items()})\n'
            '\n'
            '    # ── WITHOUT RAG baseline ──────────────────────────────────────────────────\n'
            '    print("\\n── Evaluating WITHOUT RAG (baseline) ──────────────────────")\n'
            '    ds_without  = _build_ragas_dataset(results, no_rag=True)\n'
            '    if len(ds_without) > 0:\n'
            '        if hasattr(ds_without, "samples"):\n'
            '            _ctx_without = ds_without.samples[0].retrieved_contexts[0][:80]\n'
            '        else:\n'
            '            _ctx_without = ds_without[0].get("contexts", ds_without[0].get("retrieved_contexts", [""]))[0][:80]\n'
            '    else:\n'
            '        _ctx_without = ""\n'
            '    print(f"  [WITHOUT RAG] sample context prefix: {_ctx_without!r}")\n'
            '    scores_without = _run_ragas(ds_without, llm, emb)\n'
            '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_without.items()})'
        )
        
        if target_print in source_str:
            source_str = source_str.replace(target_print, replacement_print)
        else:
            # Let's try matching with other newlines
            target_print_alt = (
                '    ds_with  = _build_ragas_dataset(results, no_rag=False)\n'
                '    # Sanity-check: confirm the two datasets differ\n'
                '    _ctx_with = ds_with[0]["contexts"][0][:80] if len(ds_with) else ""\n'
                '    print(f"  [WITH RAG]  sample context prefix: {_ctx_with!r}")\n'
                '    scores_with = _run_ragas(ds_with, llm, emb)\n'
                '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_with.items()})\n'
                '\n'
                '    # ── WITHOUT RAG baseline ──────────────────────────────────────────────────\n'
                '    print("\\n── Evaluating WITHOUT RAG (baseline) ──────────────────────")\n'
                '    ds_without  = _build_ragas_dataset(results, no_rag=True)\n'
                '    _ctx_without = ds_without[0]["contexts"][0][:80] if len(ds_without) else ""\n'
                '    print(f"  [WITHOUT RAG] sample context prefix: {_ctx_without!r}")\n'
                '    scores_without = _run_ragas(ds_without, llm, emb)\n'
                '    print("  Scores:", {k: f"{v:.4f}" for k, v in scores_without.items()})'
            )
            source_str = source_str.replace(target_print_alt, replacement_print)
            
        if source_str != original_source_str:
            # Split back into lines keeping newlines
            lines = []
            current_line = ""
            for char in source_str:
                current_line += char
                if char == "\n":
                    lines.append(current_line)
                    current_line = ""
            if current_line:
                lines.append(current_line)
            cell["source"] = lines
            modified = True
            print("Successfully updated llm_eval_stage print statements in main.ipynb cell.")

if modified:
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1)
    print("main.ipynb saved successfully.")
else:
    print("No changes were needed for main.ipynb.")
