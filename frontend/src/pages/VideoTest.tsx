import { useCallback, useEffect, useRef, useState } from 'react'
import { Upload, Play, Pause, SkipBack, SkipForward } from 'lucide-react'
import StatusBadge from '../components/StatusBadge'
import JsonViewer from '../components/JsonViewer'
import { videoJob, type PersonTrack, type Detection } from '../mocks/data'

const MAX_TIME_DELTA = 0.25
const COLOR_KNOWN = '#10b981'
const COLOR_ANON = '#94a3b8'
const COLOR_NEW_ANON = '#f97316'

function getClosestDetection(person: PersonTrack, time: number): Detection | null {
  let best: Detection | null = null
  let bestDelta = Infinity
  for (const d of person.detections) {
    const delta = Math.abs(d.timestamp - time)
    if (delta < bestDelta) {
      bestDelta = delta
      best = d
    }
  }
  return bestDelta <= MAX_TIME_DELTA ? best : null
}

function VideoTest() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [job] = useState(videoJob)
  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)
  const [playing, setPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const [threshold, setThreshold] = useState(0.75)

  const drawOverlay = useCallback(() => {
    const video = videoRef.current
    const canvas = canvasRef.current
    if (!video || !canvas) return

    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const displayWidth = canvas.clientWidth
    const displayHeight = canvas.clientHeight
    if (canvas.width !== displayWidth || canvas.height !== displayHeight) {
      canvas.width = displayWidth
      canvas.height = displayHeight
    }

    ctx.clearRect(0, 0, displayWidth, displayHeight)

    const scaleX = displayWidth / job.width
    const scaleY = displayHeight / job.height
    const time = video.currentTime

    job.persons.forEach((person) => {
      const detection = getClosestDetection(person, time)
      if (!detection) return
      if (detection.confidence < threshold) return

      const bb = detection.boundingBox
      const x = bb.x * scaleX
      const y = bb.y * scaleY
      const w = bb.width * scaleX
      const h = bb.height * scaleY

      const color =
        person.status === 'known'
          ? COLOR_KNOWN
          : person.status === 'anonymous'
          ? COLOR_ANON
          : COLOR_NEW_ANON

      ctx.strokeStyle = color
      ctx.lineWidth = selectedTrackId && selectedTrackId !== person.trackId ? 2 : 3
      ctx.strokeRect(x, y, w, h)

      if (!selectedTrackId || selectedTrackId === person.trackId) {
        const label = `${person.name || person.faceId} • ${(detection.confidence * 100).toFixed(0)}%`
        ctx.fillStyle = color
        const padding = 4
        const textWidth = ctx.measureText(label).width + padding * 2
        ctx.fillRect(x, y - 20, textWidth, 20)
        ctx.fillStyle = '#ffffff'
        ctx.font = '12px sans-serif'
        ctx.fillText(label, x + padding, y - 5)
      }
    })
  }, [job, selectedTrackId, threshold])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return

    const onTime = () => setCurrentTime(video.currentTime)
    video.addEventListener('timeupdate', onTime)
    video.addEventListener('timeupdate', drawOverlay)
    window.addEventListener('resize', drawOverlay)
    return () => {
      video.removeEventListener('timeupdate', onTime)
      video.removeEventListener('timeupdate', drawOverlay)
      window.removeEventListener('resize', drawOverlay)
    }
  }, [drawOverlay])

  useEffect(() => {
    drawOverlay()
  }, [drawOverlay])

  const togglePlay = () => {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play()
      setPlaying(true)
    } else {
      video.pause()
      setPlaying(false)
    }
  }

  const seekTo = (time: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.min(time, video.duration || time)
  }

  const stepFrame = (dir: number) => {
    const video = videoRef.current
    if (!video) return
    video.currentTime = Math.max(0, video.currentTime + dir * 0.1)
  }

  const selectTrack = (person: PersonTrack) => {
    setSelectedTrackId(person.trackId)
    seekTo(person.firstSeen)
  }

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Video Test</h2>
        <p className="text-sm text-[var(--color-muted)]">Validate recognition with bounding box overlay</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <div className="card p-4">
            <div className="flex flex-wrap items-center justify-between gap-3">
              <div>
                <p className="text-sm font-medium text-[var(--color-foreground)]">Job {job.jobId}</p>
                <p className="text-xs text-[var(--color-dim)]">
                  {job.processedFrames}/{job.totalFrames} frames • {job.duration}s • {job.width}×{job.height}
                </p>
              </div>
              <StatusBadge status={job.status} />
            </div>
            <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[var(--color-elevated)]">
              <div
                className="h-full rounded-full bg-[var(--color-primary)] transition-all"
                style={{ width: `${job.progress}%` }}
              />
            </div>
          </div>

          <div className="relative overflow-hidden rounded-xl border border-[var(--color-border)] bg-black shadow-lg">
            <video
              ref={videoRef}
              src={job.videoUrl}
              className="w-full"
              controls={false}
              preload="metadata"
              crossOrigin="anonymous"
            />
            <canvas
              ref={canvasRef}
              className="pointer-events-none absolute inset-0 h-full w-full"
            />
          </div>

          <div className="card flex flex-wrap items-center gap-3 p-3">
            <button
              type="button"
              onClick={togglePlay}
              className="btn-primary p-2"
            >
              {playing ? <Pause size={18} /> : <Play size={18} />}
            </button>
            <button
              type="button"
              onClick={() => stepFrame(-1)}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-elevated)] p-2 text-[var(--color-muted)] hover:bg-[var(--color-border-light)]"
            >
              <SkipBack size={18} />
            </button>
            <button
              type="button"
              onClick={() => stepFrame(1)}
              className="rounded-lg border border-[var(--color-border)] bg-[var(--color-elevated)] p-2 text-[var(--color-muted)] hover:bg-[var(--color-border-light)]"
            >
              <SkipForward size={18} />
            </button>
            <div className="ml-auto font-mono text-sm text-[var(--color-foreground)]">
              {currentTime.toFixed(2)} / {job.duration.toFixed(1)}s
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Person Tracks</h3>
            <div className="mt-4 space-y-3">
              {job.persons.map((person) => (
                <button
                  key={person.trackId}
                  type="button"
                  onClick={() => selectTrack(person)}
                  className={`w-full rounded-lg border p-3 text-left transition-all ${
                    selectedTrackId === person.trackId
                      ? 'border-[var(--color-primary)] bg-[var(--color-primary)]/10'
                      : 'border-[var(--color-border)] bg-[var(--color-bg)] hover:border-[var(--color-border-light)]'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                      <div
                        className="h-10 w-10 rounded-full"
                        style={{
                          backgroundColor:
                            person.status === 'known'
                              ? `${COLOR_KNOWN}20`
                              : person.status === 'anonymous'
                              ? `${COLOR_ANON}25`
                              : `${COLOR_NEW_ANON}20`,
                          border: `2px solid ${
                            person.status === 'known'
                              ? COLOR_KNOWN
                              : person.status === 'anonymous'
                              ? COLOR_ANON
                              : COLOR_NEW_ANON
                          }`,
                        }}
                      />
                      <div>
                        <p className="text-sm font-semibold text-[var(--color-foreground)]">
                          {person.name || person.faceId}
                        </p>
                        <p className="text-xs text-[var(--color-muted)]">
                          {person.firstSeen}s – {person.lastSeen}s • {(person.confidence * 100).toFixed(0)}%
                        </p>
                      </div>
                    </div>
                    <StatusBadge status={person.status} />
                  </div>
                  <div className="mt-3 h-2 w-full overflow-hidden rounded-full bg-[var(--color-elevated)]">
                    <div
                      className="h-full rounded-full bg-[var(--color-primary)]"
                      style={{
                        marginLeft: `${(person.firstSeen / job.duration) * 100}%`,
                        width: `${((person.lastSeen - person.firstSeen) / job.duration) * 100}%`,
                      }}
                    />
                  </div>
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Configuration</h3>
            <div className="mt-4 space-y-4">
              <div>
                <label htmlFor="sample" className="text-xs font-medium uppercase tracking-wider text-[var(--color-dim)]">
                  Sampling rate
                </label>
                <select
                  id="sample"
                  className="mt-2 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 py-2 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)]"
                >
                  <option>Every 10th frame</option>
                  <option>Every 5th frame</option>
                  <option>1 fps</option>
                </select>
              </div>
              <div>
                <label htmlFor="vthreshold" className="text-xs font-medium uppercase tracking-wider text-[var(--color-dim)]">
                  Confidence threshold
                </label>
                <input
                  id="vthreshold"
                  type="range"
                  min={0}
                  max={1}
                  step={0.05}
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="mt-2 w-full accent-[var(--color-primary)]"
                />
                <p className="text-right text-xs text-[var(--color-muted)]">{threshold.toFixed(2)}</p>
              </div>
            </div>
          </div>

          <div className="card flex flex-col items-center justify-center p-6 text-center">
            <Upload className="h-8 w-8 text-[var(--color-dim)]" />
            <p className="mt-2 text-sm font-medium text-[var(--color-foreground)]">Upload new video</p>
            <p className="text-xs text-[var(--color-dim)]">MP4, MOV up to 100 MB</p>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Raw JSON</h3>
            <div className="mt-4">
              <JsonViewer data={job} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default VideoTest
