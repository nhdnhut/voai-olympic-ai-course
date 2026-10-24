"""
builder/utils.py
Helper functions to generate pedagogical Jupyter notebooks adhering to the official syllabus.
"""

import nbformat as nbf
import os
import json

def create_notebook(
    module_id: str,
    module_name: str,
    session_id: str,
    topic: str,
    objectives: list,
    prerequisites: list,
    estimated_time: str,
    session_output: str,
    competition_relevance: str,
    syllabus_traceability: dict,
    why_content: str,
    concept_content: str,
    visual_intuition_content: str,
    code_cells: list,  # list of tuples: (code_str, walkthrough_markdown)
    practice_content: str,
    voai_connection_content: str,
    debugging_content: str,
    knowledge_check_content: str,
    session_summary_content: str,
    next_session_content: str,
    module_assessment: str = None
) -> nbf.v4.new_notebook:
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "language_info": {
            "name": "python",
            "version": "3.10"
        },
        "kernelspec": {
            "name": "python3",
            "display_name": "Python 3"
        }
    }

    # 0. Session Header & Metadata
    obj_str = "\n".join([f"- {obj}" for obj in objectives])
    prereq_str = "\n".join([f"- {p}" for p in prerequisites])
    
    header_md = f"""# [{module_id}] {module_name}
## Buổi {session_id}: {topic}

---

### Metadata Khóa Học
* **Module:** {module_id} — {module_name}
* **Buổi:** {session_id}
* **Thời lượng dự kiến:** {estimated_time}
* **Sản phẩm đầu ra (Session Output):** {session_output}
* **Mức độ liên quan VOAI / Olympic AI:** {competition_relevance}

#### Mục tiêu học tập (Learning Objectives)
{obj_str}

#### Điều kiện tiên quyết (Prerequisites)
{prereq_str}

---
"""
    nb.cells.append(nbf.v4.new_markdown_cell(header_md))

    # 1. WHY?
    why_md = f"""## 1. WHY? — TẠI SAO PHẢI HỌC CHỦ ĐỀ NÀY?

{why_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(why_md))

    # 2. CONCEPT
    concept_md = f"""## 2. CONCEPT — NỀN TẢNG LÝ THUYẾT & TRỰC GIÁC

{concept_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(concept_md))

    # 3. VISUAL INTUITION
    visual_md = f"""## 3. VISUAL INTUITION — TRỰC QUAN HÓA KIẾN THỨC

{visual_intuition_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(visual_md))

    # 4 & 5. CODE & CODE WALKTHROUGH
    nb.cells.append(nbf.v4.new_markdown_cell("## 4. CODE IMPLEMENTATION & 5. WALKTHROUGH\n\n*Toàn bộ code dưới đây được thiết kế thực tế, có thể chạy độc lập không phụ thuộc tải dữ liệu bên ngoài.*"))

    for i, (code_text, walkthrough_text) in enumerate(code_cells, start=1):
        nb.cells.append(nbf.v4.new_code_cell(code_text.strip()))
        walkthrough_md = f"""### 🔍 Code Walkthrough (Khối {i})
{walkthrough_text.strip()}
"""
        nb.cells.append(nbf.v4.new_markdown_cell(walkthrough_md))

    # 6. PRACTICE
    practice_md = f"""## 6. PRACTICE — BÀI TẬP THỰC HÀNH

{practice_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(practice_md))

    # 7. VOAI CONNECTION
    voai_md = f"""## 7. VOAI CONNECTION — CHIẾN LƯỢC PHÒNG THI OLYMPIC

{voai_connection_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(voai_md))

    # 8. DEBUGGING
    debug_md = f"""## 8. DEBUGGING CHALLENGE — BẮT LỖI THỰC CHIẾN

{debugging_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(debug_md))

    # 9. KNOWLEDGE CHECK
    quiz_md = f"""## 9. KNOWLEDGE CHECK — KIỂM TRA ĐỘ HIỂU BẢN CHẤT

{knowledge_check_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(quiz_md))

    # 10. SESSION SUMMARY
    summary_md = f"""## 10. SESSION SUMMARY — TỔNG KẾT BUỔI HỌC

{session_summary_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(summary_md))

    # 11. NEXT SESSION
    next_md = f"""## 11. NEXT SESSION — BƯỚC ĐI TIẾP THEO

{next_session_content}
"""
    nb.cells.append(nbf.v4.new_markdown_cell(next_md))

    # Module Assessment if provided
    if module_assessment:
        nb.cells.append(nbf.v4.new_markdown_cell(module_assessment))

    # Syllabus Traceability
    trace_items = "\n".join([f"- {item} ✓" for item in syllabus_traceability.get("items", [])])
    trace_md = f"""---

## 🎯 Syllabus Traceability (Đối chiếu Syllabus)

Buổi học này đóng góp trực tiếp vào mục tiêu của **{module_id}**:
{trace_items}

* **Session Deliverable:** {syllabus_traceability.get("deliverable", session_output)}
* **Đảm bảo chuẩn Syllabus Lock:** Nội dung nằm đúng chuẩn 100%, không lan man, giải quyết trực tiếp năng lực cốt lõi.
"""
    nb.cells.append(nbf.v4.new_markdown_cell(trace_md))

    return nb

def save_notebook(nb, filepath: str):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"[OK] Saved {filepath}")
