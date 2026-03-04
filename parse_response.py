import re
from typing import List, Dict, Any, Optional

# 兼容：
# "Step 1: ..."
# "# Step 1: ..."
# "### Step 1: ..."
STEP_HEADER_RE = re.compile(
    r"(?:^|\n)\s*(?:#+\s*)?Step\s*(\d+)\s*:\s*[^\n]*\n",
    re.IGNORECASE
)

def clean_markdown_list(text: str) -> str:
    """把 markdown/bullet 清洗成可读的一段文本（保留信息，不强行结构化）"""
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # 去掉 bullet
        line = re.sub(r"^[-*]\s*", "", line)
        # 去掉粗体
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        lines.append(line)
    return " ".join(lines).strip()

def extract_decisions(step4_text: str) -> List[str]:
    """
    从 Step 4 抽取 decision list。
    兼容：
    - "G. Adjust existing component parameters"
    - "- **B. ...**"
    - "Primary Decisions: - **B...** - **D...**"
    - 只给一句话的情况
    """
    decisions: List[str] = []

    for raw in step4_text.splitlines():
        line = raw.strip()
        if not line:
            continue

        # 去掉 bullet 和加粗
        line = re.sub(r"^[-*]\s*", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)

        # 匹配 "G. xxx"
        m = re.match(r"^[A-Z]\.\s*(.+)$", line)
        if m:
            decisions.append(m.group(1).strip())
            continue

    if not decisions:
        # fallback：整段当一个 decision（再做清洗）
        t = clean_markdown_list(step4_text)
        if t:
            # 也尝试去掉可能的 "Primary Decisions:" 前缀
            t = re.sub(r"^Primary Decisions\s*:?\s*", "", t, flags=re.I).strip()
            if t:
                decisions = [t]

    return decisions

def parse_llm_response(text: str) -> List[Dict[str, Any]]:
    """
    统一解析 Step 1~5，输出你前端需要的 model_response item 列表。
    """
    matches = list(STEP_HEADER_RE.finditer(text))
    steps: Dict[int, str] = {}

    for i, m in enumerate(matches):
        step_id = int(m.group(1))
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        steps[step_id] = text[start:end].strip()

    # Step 1: judgement 通常是第一行 "Bad Fit/Good Fit"（可能带 **）
    s1 = steps.get(1, "").strip()
    s1_lines = [ln.strip() for ln in s1.splitlines() if ln.strip()]
    overall = ""
    if s1_lines:
        overall = re.sub(r"\*\*(.*?)\*\*", r"\1", s1_lines[0]).strip()
    else:
        overall = re.sub(r"\*\*(.*?)\*\*", r"\1", s1).strip()

    stat_issues = clean_markdown_list(steps.get(2, ""))
    image_issues = clean_markdown_list(steps.get(3, ""))
    decisions = extract_decisions(steps.get(4, ""))
    reasons = clean_markdown_list(steps.get(5, ""))

    return [
        {"key": "overall_judgement", "title": "Overall Judgement", "content": overall},
        {"key": "observed_statistical_issues", "title": "Observed Statistical Issues", "content": stat_issues},
        {"key": "observed_image_issues", "title": "Observed Image Issues", "content": image_issues},
        {"key": "next_decision", "title": "Next-step Decision", "content": decisions},
        {"key": "decision_reasons", "title": "Decision Reasons", "content": reasons},
    ]
