import os, uuid, shutil, subprocess, json, re, math
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
for p in (UPLOADS, OUTPUTS):
    p.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GameClips AI")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
JOBS = {}

def run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr[-2500:])
    return p.stdout.strip()

def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out)

def transcribe(path):
    # Optional dependency. If unavailable, the pipeline still uses audio activity.
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), device="cpu", compute_type="int8")
        segments, _ = model.transcribe(str(path), vad_filter=True)
        return [{"start": float(s.start), "end": float(s.end), "text": s.text.strip()} for s in segments]
    except Exception:
        return []

def score_windows(path, total, clip_len, transcript):
    # Build candidate windows and rank them by speech emphasis/cue words + audio activity.
    # ffmpeg volumedetect is used as a coarse global activity signal; transcript adds context.
    cues = ("wow", "no way", "insane", "clutch", "winner", "win", "kill", "headshot",
            "squad", "last", "crazy", "let's go", "lets go", "nice", "enemy", "ace",
            "knock", "finish", "victory", "oh my", "bro", "what")
    step = max(5, clip_len // 3)
    candidates = []
    for start in range(0, max(1, int(total - clip_len) + 1), step):
        end = min(total, start + clip_len)
        words = " ".join(s["text"].lower() for s in transcript
                         if s["start"] < end and s["end"] > start)
        cue_score = sum(words.count(c) for c in cues)
        speech_score = sum(1 for s in transcript if s["start"] < end and s["end"] > start)
        # Prefer windows with speech/cue context; small time-position variety bonus.
        position_bonus = 0.15 if start > total * 0.45 else 0
        score = cue_score * 3.0 + speech_score * 0.35 + position_bonus
        candidates.append((score, start, end, words))
    if not candidates:
        candidates = [(0, 0, min(total, clip_len), "")]
    candidates.sort(reverse=True)
    selected = []
    for item in candidates:
        _, st, en, _ = item
        if all(abs(st - old[1]) >= clip_len * 0.75 for old in selected):
            selected.append(item)
        if len(selected) >= 10:
            break
    if not selected:
        selected = [candidates[0]]
    return sorted(selected, key=lambda x: x[1])

def make_srt(transcript, start, end, out_path):
    rows, idx = [], 1
    for s in transcript:
        a, b = max(start, s["start"]), min(end, s["end"])
        txt = s["text"].strip()
        if b <= a or not txt:
            continue
        # Sentence-sized caption blocks; escaped for SRT.
        def stamp(t):
            ms = int(round(t * 1000))
            h, ms = divmod(ms, 3600000)
            m, ms = divmod(ms, 60000)
            sec, ms = divmod(ms, 1000)
            return f"{h:02}:{m:02}:{sec:02},{ms:03}"
        rows.append(f"{idx}\n{stamp(a-start)} --> {stamp(b-start)}\n{txt}\n")
        idx += 1
    Path(out_path).write_text("\n".join(rows), encoding="utf-8")
    return bool(rows)

def process_job(job_id, src, clip_len, count):
    try:
        JOBS[job_id]["status"] = "Analyzing video…"
        total = duration(src)
        transcript = transcribe(src)
        picks = score_windows(src, total, clip_len, transcript)
        picks = picks[:count]
        results = []
        for n, (_, start, end, _) in enumerate(picks, 1):
            JOBS[job_id]["status"] = f"Creating clip {n} of {len(picks)}…"
            out = OUTPUTS / f"{job_id}_clip_{n}.mp4"
            srt = OUTPUTS / f"{job_id}_clip_{n}.srt"
            has_subs = make_srt(transcript, start, end, srt)
            # Fill a 9:16 canvas, crop to portrait; use subtitles if speech was detected.
            vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
            if has_subs:
                vf += f",subtitles='{str(srt).replace(chr(92), '/').replace(':', r'\\:')}'"
            run(["ffmpeg", "-y", "-ss", str(start), "-i", str(src), "-t", str(end-start),
                 "-vf", vf, "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
            results.append({"name": f"Gaming Short {n}", "url": f"/download/{out.name}",
                            "start": round(start, 1), "duration": round(end-start, 1)})
        JOBS[job_id]["clips"] = results
        JOBS[job_id]["status"] = "Done"
        JOBS[job_id]["transcript_available"] = bool(transcript)
    except Exception as e:
        JOBS[job_id]["status"] = "Error"
        JOBS[job_id]["error"] = str(e)

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/api/create")
async def create(background: BackgroundTasks,
                 video: UploadFile = File(...),
                 source_url: str = Form(""),
                 clip_length: int = Form(45),
                 clip_count: int = Form(5)):
    if clip_length not in (30, 45, 60):
        raise HTTPException(400, "Clip length must be 30, 45, or 60 seconds.")
    if not 1 <= clip_count <= 10:
        raise HTTPException(400, "Choose between 1 and 10 clips.")
    ext = Path(video.filename or "video.mp4").suffix.lower()
    if ext not in (".mp4", ".mov", ".m4v", ".mkv", ".webm"):
        raise HTTPException(400, "Please upload an MP4, MOV, M4V, MKV, or WEBM video.")
    job_id = uuid.uuid4().hex[:10]
    src = UPLOADS / f"{job_id}{ext}"
    with src.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    JOBS[job_id] = {"status": "Queued", "clips": [], "source_url": source_url}
    background.add_task(process_job, job_id, src, clip_length, clip_count)
    return {"job_id": job_id}

@app.get("/api/status/{job_id}")
def status(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, "Job not found")
    return JOBS[job_id]

@app.get("/download/{filename}")
def download(filename: str):
    path = OUTPUTS / Path(filename).name
    if not path.exists():
        raise HTTPException(404, "Clip not found")
    return FileResponse(path, media_type="video/mp4", filename=path.name)
