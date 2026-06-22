# API Documentation

Base URL: `http://localhost:8000`  
Interactive docs: `http://localhost:8000/docs`

---

## GET /health

Check if the API is running and the model is loaded.

**Response:**
```json
{
  "status": "ok",
  "model_loaded": true,
  "model_name": "densenet121",
  "device": "cpu",
  "version": "1.0.0"
}
```

---

## POST /predict

Upload a chest X-ray image and get a binary prediction.

**Request:** `multipart/form-data`  
**Field:** `file` — JPEG or PNG image, max 10 MB

**Response:**
```json
{
  "predicted_class": "PNEUMONIA",
  "predicted_index": 1,
  "confidence": 0.9732,
  "probabilities": {
    "NORMAL": 0.0268,
    "PNEUMONIA": 0.9732
  },
  "model_name": "densenet121",
  "image_size": [224, 224]
}
```

---

## POST /explain/gradcam

Generate a Grad-CAM heatmap for the uploaded image.

**Request:** `multipart/form-data`  
**Field:** `file` — JPEG or PNG image

**Response:**
```json
{
  "predicted_class": "PNEUMONIA",
  "confidence": 0.9732,
  "original_b64": "<base64 PNG>",
  "heatmap_b64":  "<base64 PNG>",
  "overlay_b64":  "<base64 PNG>",
  "explanation":  "Grad-CAM: Model predicted PNEUMONIA with 97.3% confidence..."
}
```

Render images with: `<img src="data:image/png;base64,{overlay_b64}" />`

---

## POST /explain/lime

Generate LIME superpixel explanations for the uploaded image.

**Request:** `multipart/form-data`  
**Field:** `file` — JPEG or PNG image

**Response:**
```json
{
  "available": true,
  "predicted_class": "PNEUMONIA",
  "confidence": 0.9732,
  "positive_b64": "<base64 PNG>",
  "negative_b64": "<base64 PNG>",
  "all_b64":      "<base64 PNG>",
  "explanation":  "LIME: Model predicted PNEUMONIA..."
}
```

If LIME is not installed, `available` will be `false` with an `error` message.

---

## Error Responses

| Code | Meaning |
|------|---------|
| 400  | Invalid image file |
| 422  | Unsupported file type or file too large |
| 500  | Model inference error |
