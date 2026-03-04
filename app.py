#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import json
import time
import math
import base64
import shutil
import tempfile
import threading
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, request, jsonify
from PIL import Image

# 你自己的解析器（保持不变）
from parse_response import parse_llm_response


# =========================
# Config (env-driven)
# =========================
TZ = timezone(timedelta(hours=8))

API_BASE_URL = os.getenv("API_BASE_URL", "https://api2.road2all.com/v1/chat/completions")
API_KEY = os.getenv("API_KEY")  # 必须通过环境变量注入
MODEL_NAME = os.getenv("MODEL_NAME", "claude-3-7-sonnet-20250219")

HOST = os.getenv("HOST", "10.15.48.208")
PORT = int(os.getenv("PORT", "5000"))

MAX_ATTEMPTS = int(os.getenv("MAX_ATTEMPTS", "3"))
MAX_TOTAL_SECONDS = int(os.getenv("MAX_TOTAL_SECONDS", "60"))
RETRY_BACKOFF_SECONDS = float(os.getenv("RETRY_BACKOFF_SECONDS", "2"))

# 图片大小限制（默认 3MB，你原注释写 5MB 但实际是 3MB）
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(3 * 1024 * 1024)))

# 日志目录
LOG_ROOT = os.getenv("SERVICE_LOG_PATH", "service")
LOG_DIR = os.path.join(LOG_ROOT, "model_log")
os.makedirs(LOG_DIR, exist_ok=True)

SERVICE_START_TIME = datetime.now(TZ)
LOG_FILENAME = f"log_{SERVICE_START_TIME.strftime('%Y%m%d_%H%M%S')}.jsonl"
LOG_PATH = os.path.join(LOG_DIR, LOG_FILENAME)

_log_lock = threading.Lock()


# =========================
# Response schema
# =========================
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


@dataclass
class ServiceResponse:
    status: ResponseStatus
    model_response: Optional[List[Dict[str, Any]]] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"status": self.status.value}
        if self.request_id:
            d["request_id"] = self.request_id
        if self.status == ResponseStatus.SUCCESS:
            d["model_response"] = self.model_response
        else:
            d["error_message"] = self.error_message or "Unknown error"
        return d


def create_success_response(model_response: List[Dict[str, Any]], request_id: str) -> ServiceResponse:
    return ServiceResponse(status=ResponseStatus.SUCCESS, model_response=model_response, request_id=request_id)


def create_error_response(error_message: str, request_id: str) -> ServiceResponse:
    return ServiceResponse(status=ResponseStatus.ERROR, error_message=error_message, request_id=request_id)


# =========================
# Logging (JSONL)
# =========================
def append_service_log(
    galaxy_id: str,
    round_id: int,
    model_response: List[Dict[str, Any]],
    request_id: str,
):
    record = {
        "request_id": request_id,
        "galaxy_id": galaxy_id,
        "round_id": round_id,
        "time": datetime.now(TZ).isoformat(),
        "model_response": model_response,
    }
    line = json.dumps(record, ensure_ascii=False)
    with _log_lock:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")


# =========================
# Prompt builder (keep your original)
# =========================
def build_single_fit_eval_prompt(text: str, round_id: int, fitting_mode: str) -> str:
    SED_IMAGE_BLOCK = """
The second input image illustrates the fitting performance of the SED model, particularly highlighting the black and red data points shown in the chart.
""".strip()

    SED_INSPECTION_BLOCK = """
SED plot:
- Check whether model points are close to observed points overall
- Check whether systematic offsets exist across wavelengths
""".strip()

    BASE_PROMPT = """
You are an astronomer experienced in galaxy component decomposition. Your task is to evaluate whether the current fitting result is physically reasonable.

# Workflow Context - Four-Stage Strategy
The GalfitS fitting process follows a sequential four-stage approach. You MUST identify which phase the user is currently in before providing recommendations:

1. Stage 1 (Image Fitting): Fit multi-band images WITHOUT SED constraints.
2. Stage 2 (Mass Guess): Single Sérsic Mass Estimation
3. Stage 3 (SED Fitting): Fix image parameters from Stage 1; fit ONLY SED parameters.
4. Stage 4 (Image-SED Fitting): Use results from Stage 1 & 3 as initial guesses; relax ALL parameters.

You are given the fitting result of a galaxy from Round {round_id}. The fitting mode of this round is {fitting_mode}.

The input image contains three panels:
- Left: original cutout image
- Middle: fitted model image
- Right: residual image (data - model)

{SED_IMAGE_OPTION}

You are also given the fitted parameter file below.

====================
FITTED PARAMETERS
====================
{text}
====================

Please strictly follow the output format below.

Step 1: Overall Judgment
- "Good Fit" or "Bad Fit"

Step 2: Observed Statistical Problems

Step 3: Observed Image Problems

Step 4: Next-Step Decision
(choose actions based on fitting_mode constraints)

Step 5: Reasons for Decision
""".strip()

    use_sed = fitting_mode not in ["Image Fitting", "Mass Guess"]

    return BASE_PROMPT.format(
        round_id=round_id,
        fitting_mode=fitting_mode,
        text=text,
        SED_IMAGE_OPTION=SED_IMAGE_BLOCK if use_sed else "",
    )


# =========================
# Image utils
# =========================
def compress_png_bytes(image_bytes: bytes, max_bytes: int = MAX_IMAGE_BYTES) -> bytes:
    """If > max_bytes, resize down until <= max_bytes. Return PNG bytes."""
    if len(image_bytes) <= max_bytes:
        return image_bytes

    img = Image.open(io.BytesIO(image_bytes))

    ratio = math.sqrt(max_bytes / len(image_bytes)) * 0.95
    new_w = max(1, int(img.width * ratio))
    new_h = max(1, int(img.height * ratio))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    out = buf.getvalue()

    # if still too big, iteratively shrink
    while len(out) > max_bytes and img.width > 1 and img.height > 1:
        new_w = max(1, int(img.width * 0.9))
        new_h = max(1, int(img.height * 0.9))
        img = img.resize((new_w, new_h), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=True)
        out = buf.getvalue()

    return out


def decode_base64_image_to_file(img_b64: str, out_path: str) -> None:
    if img_b64.startswith("data:"):
        img_b64 = img_b64.split(",", 1)[1]
    raw = base64.b64decode(img_b64)
    raw = compress_png_bytes(raw, MAX_IMAGE_BYTES)
    with open(out_path, "wb") as f:
        f.write(raw)


# =========================
# Model client
# =========================
def get_model_response(question: str, image_paths: List[str], temperature: float = 0.1, timeout: int = 45) -> str:
    if not API_KEY:
        raise RuntimeError("Missing API_KEY env var")

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    content: List[Dict[str, Any]] = [{"type": "text", "text": question}]

    for p in image_paths:
        with open(p, "rb") as f:
            b = f.read()
        b64 = base64.b64encode(b).decode("utf-8")
        # 你网关这里按 image_url 的 data URI 传；保持与原逻辑一致
        content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})

    payload = {
        "model": MODEL_NAME,
        "temperature": temperature,
        "messages": [{"role": "user", "content": content}],
    }

    resp = requests.post(API_BASE_URL, headers=headers, json=payload, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"Upstream API error {resp.status_code}: {resp.text}")

    j = resp.json()
    return j.get("choices", [{}])[0].get("message", {}).get("content", "") or ""


# =========================
# Request validation
# =========================
def validate_body(body: Dict[str, Any]) -> Optional[str]:
    if not isinstance(body, dict):
        return "Body must be a JSON object"

    if not (body.get("galaxy_id") or body.get("object_id")):
        return "Missing galaxy_id"

    if body.get("round_id") is None:
        return "Missing round_id"

    if not body.get("fitting_mode"):
        return "Missing fitting_mode"

    images = body.get("images")
    if not isinstance(images, list) or len(images) == 0:
        return "Missing images (base64 list)"

    return None


# =========================
# Core service logic
# =========================
def evaluate_once(body: Dict[str, Any], request_id: str) -> ServiceResponse:
    object_id = str(body.get("galaxy_id") or body.get("object_id"))
    round_id = int(body.get("round_id"))
    fitting_mode = str(body.get("fitting_mode"))
    summary_text = body.get("gssummary") or body.get("summary") or ""
    images: List[str] = body.get("images") or []

    tmpdir = tempfile.mkdtemp(prefix="model_response_")
    image_paths: List[str] = []

    try:
        for idx, imgdata in enumerate(images):
            if not imgdata:
                continue
            img_path = os.path.join(tmpdir, f"{object_id}_img{idx}.png")
            decode_base64_image_to_file(imgdata, img_path)
            image_paths.append(img_path)

        if not image_paths:
            return create_error_response("No valid images provided.", request_id)

        prompt = build_single_fit_eval_prompt(
            text=summary_text,
            round_id=round_id,
            fitting_mode=fitting_mode,
        )

        start_time = time.time()
        attempt = 0

        while attempt < MAX_ATTEMPTS:
            elapsed = time.time() - start_time
            remaining = MAX_TOTAL_SECONDS - elapsed
            if remaining <= 0:
                break

            attempt += 1
            try:
                resp_text = get_model_response(
                    question=prompt,
                    image_paths=image_paths,
                    timeout=min(int(remaining), 45),
                )
                if not resp_text.strip():
                    time.sleep(min(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), remaining))
                    continue

                parsed_items = parse_llm_response(resp_text)

                # log
                try:
                    append_service_log(object_id, round_id, parsed_items, request_id)
                except Exception:
                    pass

                return create_success_response(parsed_items, request_id)

            except Exception as e:
                # backoff
                sleep_t = min(RETRY_BACKOFF_SECONDS * (2 ** (attempt - 1)), remaining)
                if sleep_t > 0:
                    time.sleep(sleep_t)

        return create_error_response(
            f"Model failed after {attempt} attempts in {int(time.time() - start_time)} seconds",
            request_id,
        )

    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# =========================
# Flask app
# =========================
app = Flask(__name__)


@app.route("/healthz", methods=["GET"])
def healthz():
    return jsonify({"status": "ok"}), 200


@app.route("/model/response", methods=["POST"])
def model_response():
    request_id = request.headers.get("X-Request-ID") or f"req_{int(time.time()*1000)}"

    try:
        body = request.get_json(force=True)
    except Exception:
        resp = create_error_response("Invalid JSON body", request_id)
        return jsonify(resp.to_dict()), 400

    err = validate_body(body)
    if err:
        resp = create_error_response(err, request_id)
        return jsonify(resp.to_dict()), 400

    resp = evaluate_once(body, request_id)
    code = 200 if resp.status == ResponseStatus.SUCCESS else 500
    return jsonify(resp.to_dict()), code


if __name__ == "__main__":
    print(f"Starting service on {HOST}:{PORT}")
    app.run(host=HOST, port=PORT)