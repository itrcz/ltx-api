"""
Local LTX test UI.

Run: pip install fastapi uvicorn python-multipart boto3 httpx python-dotenv
     uvicorn app:app --reload --port 8080

Reads config from ../.env (RUNPOD_API_KEY, S3_*, etc).
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path

import boto3
import httpx
from botocore.config import Config
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv(Path(__file__).parent.parent / ".env")

RUNPOD_API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ.get("LTX_ENDPOINT_ID", "d7kud62ob6wwtp")
S3_ENDPOINT = os.environ.get("S3_ENDPOINT_URL") or None
S3_REGION = os.environ.get("S3_REGION", "auto")
S3_BUCKET = os.environ["S3_BUCKET"]
S3_KEY_ID = os.environ["S3_ACCESS_KEY_ID"]
S3_SECRET = os.environ["S3_SECRET_ACCESS_KEY"]

RUNPOD_BASE = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
RUNPOD_HEADERS = {"Authorization": f"Bearer {RUNPOD_API_KEY}"}

s3 = boto3.client(
    "s3",
    endpoint_url=S3_ENDPOINT,
    region_name=S3_REGION,
    aws_access_key_id=S3_KEY_ID,
    aws_secret_access_key=S3_SECRET,
    config=Config(signature_version="s3v4", s3={"addressing_style": "virtual"}),
)

app = FastAPI()

INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>LTX test</title>
<style>
body{font-family:-apple-system,sans-serif;max-width:720px;margin:40px auto;padding:0 20px;color:#222}
h1{font-size:20px;margin:0 0 20px}
form>div{margin:12px 0}
label{display:block;font-size:13px;color:#666;margin-bottom:4px}
input[type=text],textarea,select{width:100%;padding:8px;border:1px solid #ccc;border-radius:4px;font-size:14px;font-family:inherit}
textarea{min-height:90px;resize:vertical}
.row{display:grid;grid-template-columns:repeat(4,1fr);gap:10px}
button{background:#111;color:#fff;border:0;padding:10px 16px;border-radius:4px;cursor:pointer;font-size:14px}
button:hover{background:#333}
.status{margin-top:20px;padding:16px;border:1px solid #eee;border-radius:4px;background:#fafafa;font-size:13px}
pre{white-space:pre-wrap;word-break:break-all;margin:4px 0;font-size:12px;color:#555}
video{width:100%;margin-top:12px;border-radius:4px;background:#000}
</style>
</head><body>
<h1>LTX video gen — test UI</h1>
<form id=f onsubmit="return submitJob(event)">
  <div><label>Prompt</label><textarea name=prompt required>A Russian tech blogger speaks enthusiastically about new ГРОМ LLM model.</textarea></div>
  <div><label>Negative prompt (optional)</label><textarea name=negative_prompt></textarea></div>
  <div><label>First frame image</label><input type=file name=image accept="image/*" required></div>
  <div class=row>
    <div><label>Quality</label><select name=quality><option value=draft>draft (stage1 only, 720p-ish)</option><option value=hd selected>hd (stage1+2, 960p)</option></select></div>
    <div><label>Aspect</label><select name=aspect_ratio><option value=9:16 selected>9:16</option><option value=16:9>16:9</option></select></div>
    <div><label>Duration (sec)</label><input type=number name=duration_sec min=1 max=10 value=5></div>
    <div><label>Steps</label><select name=steps><option>5</option><option selected>8</option><option>10</option><option>15</option><option>20</option></select></div>
  </div>
  <div class=row>
    <div><label>Seed</label><input type=number name=seed value=42></div>
  </div>
  <button type=submit>Generate</button>
</form>
<div id=status class=status style=display:none></div>
<script>
async function submitJob(e){
  e.preventDefault();
  const form = new FormData(e.target);
  const s = document.getElementById('status');
  s.style.display='block';
  s.innerHTML='Uploading image and submitting job…';
  const r = await fetch('/submit', {method:'POST', body: form});
  if(!r.ok){ s.innerHTML = 'Submit failed: '+await r.text(); return false; }
  const {job_id} = await r.json();
  s.innerHTML = `<div>Job <code>${job_id}</code> queued. Polling…</div><pre id=log></pre>`;
  pollStatus(job_id);
  return false;
}
async function pollStatus(job_id){
  const s = document.getElementById('status');
  const log = document.getElementById('log');
  const t0 = Date.now();
  while(true){
    const r = await fetch('/status/'+job_id);
    const j = await r.json();
    const elapsed = ((Date.now()-t0)/1000).toFixed(0);
    log.textContent = `t+${elapsed}s  status=${j.status}  progress=${JSON.stringify(j.progress ?? null)}\\n${JSON.stringify(j.output ?? j, null, 2)}`;
    if(j.status==='COMPLETED'){
      const out = j.output || {};
      s.innerHTML += `<div style="margin-top:12px">Done in ${elapsed}s.</div>`;
      if(out.video_url){
        s.innerHTML += `<video controls src="${out.video_url}"></video><div><a href="${out.video_url}" target=_blank>Open mp4</a></div>`;
      }
      s.innerHTML += `<pre>${JSON.stringify(out, null, 2)}</pre>`;
      break;
    }
    if(j.status==='FAILED' || j.status==='CANCELLED' || j.status==='TIMED_OUT'){
      s.innerHTML += `<div style="color:#b00">Failed: ${JSON.stringify(j.output||j.error||j)}</div>`;
      break;
    }
    await new Promise(r=>setTimeout(r, 2500));
  }
}
</script>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def index():
    return INDEX_HTML


@app.post("/submit")
async def submit(
    prompt: str = Form(...),
    negative_prompt: str = Form(""),
    quality: str = Form("hd"),
    aspect_ratio: str = Form("9:16"),
    duration_sec: int = Form(5),
    steps: int = Form(8),
    seed: int = Form(42),
    image: UploadFile = File(...),
):
    img_key = f"ltx-inputs/{uuid.uuid4().hex}-{image.filename}"
    body = await image.read()
    s3.put_object(Bucket=S3_BUCKET, Key=img_key, Body=body, ContentType=image.content_type or "image/png")
    first_frame_url = s3.generate_presigned_url(
        "get_object", Params={"Bucket": S3_BUCKET, "Key": img_key}, ExpiresIn=3600
    )

    payload = {
        "input": {
            "prompt": prompt,
            "negative_prompt": negative_prompt or None,
            "quality": quality,
            "aspect_ratio": aspect_ratio,
            "duration_sec": duration_sec,
            "steps": steps,
            "seed": seed,
            "first_frame_url": first_frame_url,
        }
    }
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(f"{RUNPOD_BASE}/run", headers=RUNPOD_HEADERS, json=payload)
    if r.status_code >= 400:
        raise HTTPException(status_code=502, detail=r.text)
    data = r.json()
    return {"job_id": data.get("id"), "first_frame_url": first_frame_url}


@app.get("/status/{job_id}")
async def status(job_id: str):
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{RUNPOD_BASE}/status/{job_id}", headers=RUNPOD_HEADERS)
    return JSONResponse(r.json(), status_code=r.status_code)


@app.get("/health")
async def health():
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.get(f"{RUNPOD_BASE}/health", headers=RUNPOD_HEADERS)
    return JSONResponse(r.json(), status_code=r.status_code)
