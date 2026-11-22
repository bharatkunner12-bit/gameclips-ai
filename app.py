import os
import uuid
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlparse

from fastapi import FastAPI, Form, BackgroundTasks, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
DATA = ROOT / "data"
UPLOADS = DATA / "uploads"
OUTPUTS = DATA / "outputs"
for folder in (UPLOADS, OUTPUTS):
    folder.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="GameClips AI")
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
JOBS = {}


def run(cmd):
    result = subprocess.run(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=None
    )
    if result.returncode:
        raise RuntimeError(result.stderr[-2500:])
    return result.stdout.strip()


def download_youtube(url: str, job_id: str) -> Path:
    """Download a YouTube video locally at up to 720p for quicker processing."""
    try:
        import yt_dlp
    except ImportError:
        raise RuntimeError("yt-dlp missing. Add yt-dlp to requirements.txt and redeploy.")

    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise RuntimeError("Please enter a valid YouTube video URL.")

    output_template = str(UPLOADS / f"{job_id}.%(ext)s")
    options = {
        "format": "bestvideo[height<=720]+bestaudio/best[height<=720]/best",
        "outtmpl": output_template,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "retries": 3,
        "fragment_retries": 3,
        "concurrent_fragment_downloads": 4,
        "socket_timeout": 30,
    }
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
        downloaded = Path(ydl.prepare_filename(info))
    # yt-dlp may merge the downloaded file to mp4.
    merged = downloaded.with_suffix(".mp4")
    if merged.exists():
        return merged
    if downloaded.exists():
        return downloaded
    matches = list(UPLOADS.glob(f"{job_id}.*"))
    if matches:
        return matches[0]
    raise RuntimeError("YouTube download finished, but the video file was not found.")


def video_duration(path: Path) -> float:
    value = run([
        "ffprobe", "-v", "error", "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1", str(path)
    ])
    return float(value)


def transcribe(path: Path):
    # Tiny model is much quicker on Render CPU than the previous base model.
    try:
        from faster_whisper import WhisperModel
        model_name = os.getenv("WHISPER_MODEL", "tiny")
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
        segments, _ = model.transcribe(
            str(path), vad_filter=True, beam_size=1, word_timestamps=False
        )
        return [
            {"start": float(s.start), "end": float(s.end), "text": s.text.strip()}
            for s in segments
        ]
    except Exception:
        return []


def choose_windows(total, clip_len, transcript, count):
    cues = (
        "wow", "no way", "insane", "clutch", "winner", "win", "kill",
        "headshot", "squad", "last", "crazy", "let's go", "lets go",
        "nice", "enemy", "ace", "knock", "finish", "victory", "oh my",
        "bro", "what", "unbelievable"
    )
    step = max(5, clip_len // 3)
    candidates = []
    last_start = max(0, int(total - clip_len))
    for start in range(0, last_start + 1, step):
        end = min(total, start + clip_len)
        overlapping = [
            seg for seg in transcript
            if seg["start"] < end and seg["end"] > start
        ]
        text = " ".join(seg["text"].lower() for seg in overlapping)
        cue_score = sum(text.count(cue) for cue in cues)
        speech_score = len(overlapping)
        # Mildly favor later gameplay while still ranking by speech/cues.
        position_bonus = 0.2 if start > total * 0.4 else 0
        score = cue_score * 3 + speech_score * 0.35 + position_bonus
        candidates.append((score, start, end))

    if not candidates:
        return [(0, min(total, clip_len))]

    candidates.sort(key=lambda item: item[0], reverse=True)
    selected = []
    for _, start, end in candidates:
        if all(abs(start - old[0]) >= clip_len * 0.8 for old in selected):
            selected.append((start, end))
        if len(selected) >= count:
            break

    # If no speech/cue data is found, distribute clips across the video.
    if not selected or max(item[0] for item in candidates) == 0:
        selected = []
        usable = max(0, total - clip_len)
        for i in range(min(count, max(1, int(total // clip_len)))):
            start = (usable * i / max(1, count - 1)) if count > 1 else 0
            selected.append((start, min(total, start + clip_len)))

    return sorted(selected, key=lambda item: item[0])


def write_srt(transcript, start, end, path):
    def stamp(seconds):
        ms = max(0, int(round(seconds * 1000)))
        hours, ms = divmod(ms, 3600000)
        minutes, ms = divmod(ms, 60000)
        seconds, ms = divmod(ms, 1000)
        return f"{hours:02}:{minutes:02}:{seconds:02},{ms:03}"

    lines = []
    index = 1
    for segment in transcript:
        a = max(start, segment["start"])
        b = min(end, segment["end"])
        text = segment["text"].strip().replace("-->", "â")
        if b <= a or not text:
            continue
        lines.append(f"{index}\n{stamp(a-start)} --> {stamp(b-start)}\n{text}\n")
        index += 1
    path.write_text("\n".join(lines), encoding="utf-8")
    return bool(lines)


def process_job(job_id, url, clip_len, count):
    source = None
    try:
        JOBS[job_id]["status"] = "Downloading YouTube videoâ¦"
        source = download_youtube(url, job_id)

        JOBS[job_id]["status"] = "Checking videoâ¦"
        total = video_duration(source)
        if total < 5:
            raise RuntimeError("The video is too short to create gameplay clips.")

        JOBS[job_id]["status"] = "Analyzing gameplay/audioâ¦"
        transcript = transcribe(source)
        windows = choose_windows(total, clip_len, transcript, count)
        results = []

        for number, (start, end) in enumerate(windows, 1):
            JOBS[job_id]["status"] = f"Rendering clip {number}/{len(windows)}â¦"
            output = OUTPUTS / f"{job_id}_clip_{number}.mp4"
            srt = OUTPUTS / f"{job_id}_clip_{number}.srt"
            has_subtitles = write_srt(transcript, start, end, srt)

            # Faster render: 720x1280 and ultrafast H.264 preset.
            video_filter = (
                "scale=720:1280:force_original_aspect_ratio=increase,"
                "crop=720:1280"
            )
            if has_subtitles:
                subtitle_path = str(srt).replace("\\", "/").replace(":", r"\:")
                video_filter += f",subtitles='{subtitle_path}'"

            run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-ss", str(start), "-i", str(source),
                "-t", str(max(1, end-start)),
                "-vf", video_filter,
                "-c:v", "libx264", "-preset", "ultrafast", "-crf", "26",
                "-c:a", "aac", "-b:a", "96k",
                "-threads", "2", "-movflags", "+faststart",
                str(output)
            ])
            results.append({
                "name": f"Gaming Short {number}",
                "url": f"/download/{output.name}",
                "start": round(start, 1),
                "duration": round(end-start, 1)
            })

        JOBS[job_id]["clips"] = results
        JOBS[job_id]["status"] = "Done"
        JOBS[job_id]["transcript_available"] = bool(transcript)
    except Exception as error:
        JOBS[job_id]["status"] = "Error"
        JOBS[job_id]["error"] = str(error)
    finally:
        # Keep output clips; remove the large downloaded source to save Render disk.
        if source and source.exists():
            try:
                source.unlink()
            except OSError:
                pass


@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")


@app.post("/api/create")
async def create(
    background: BackgroundTasks,
    source_url: str = Form(...),
    clip_length: int = Form(45),
    clip_count: int = Form(10),
):
    source_url = source_url.strip()
    if not source_url:
        raise HTTPException(400, "Paste a YouTube video link.")
    if clip_length not in (30, 45, 60):
        raise HTTPException(400, "Clip length must be 30, 45, or 60 seconds.")
    if not 1 <= clip_count <= 15:
        raise HTTPException(400, "Choose between 1 and 15 clips.")

    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {
        "status": "Queued",
        "clips": [],
        "source_url": source_url,
        "error": None
    }
    background.add_task(process_job, job_id, source_url, clip_length, clip_count)
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
