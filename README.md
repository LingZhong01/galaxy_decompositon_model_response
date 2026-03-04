# Galaxy Fitting Evaluation Service

A lightweight HTTP service for evaluating galaxy fitting results using a multimodal model.

The service receives galaxy fitting images and parameter summaries, and returns a structured evaluation result.

The service is implemented as a **single script (`app.py`)**.

---

# 1. Requirements

Python ≥ 3.9

Install dependencies:

```bash
pip install flask requests pillow gunicorn
```

---

# 2. Environment Variables

Before starting the service, configure the model API key.

```bash
export API_KEY="your_api_key"
```

Optional configuration:

```bash
export MODEL_NAME="claude-3-7-sonnet-20250219"
export PORT=5000
export HOST=0.0.0.0
```

---

# 3. Start the Service

### Development mode

```bash
python app.py
```

Service will start at:

```
http://localhost:5000
```

### Production mode (recommended)

```bash
gunicorn -w 2 -k gthread -t 120 -b 0.0.0.0:5000 app:app
```

---

# 4. API

## Health Check

```
GET /healthz
```

Response:

```json
{
  "status": "ok"
}
```

---

## Evaluate Fitting Result

```
POST /model/response
```

### Request Example

```json
{
  "galaxy_id": "41926",
  "round_id": 2,
  "fitting_mode": "Image Fitting",
  "gssummary": "fitted parameter file content",
  "images": [
    "base64_image_1",
    "base64_image_2"
  ]
}
```

### Fields

| Field        | Description           |
| ------------ | --------------------- |
| galaxy_id    | galaxy identifier     |
| round_id     | fitting round         |
| fitting_mode | fitting mode          |
| gssummary    | parameter file text   |
| images       | base64 encoded images |

---


# 5. Logs

Service logs are written to:

```
SERVICE_LOG_PATH/model_log/
```

Each request will generate one JSON log entry containing the model evaluation result.

---
