import { cn } from '@/lib/utils'
import {
  calculateContainedVideoRect,
  formatMediaTimeSeconds,
  selectVisibleDetectionsAtTime,
  stableTrackColor,
} from '@/lib/video'
import { Film, Pause, Play } from 'lucide-react'
import { forwardRef, useCallback, useEffect, useRef, useState } from 'react'
import type { OverlayDetection, OverlayFrame, VideoAppearanceEntry, VideoPersonSummary } from '@/api/types'

const FONT = '500 12px system-ui, sans-serif'
const BOX_LINE_NORMAL = 2
const BOX_LINE_SELECTED = 3
const LABEL_PADDING = 4

export interface VideoOverlayPlayerProps {
  src: string | undefined
  fileName?: string
  className?: string
  displayWidth: number
  displayHeight: number
  rotationDegrees?: number
  durationSeconds: number
  frames: OverlayFrame[]
  people: VideoPersonSummary[]
  appearances: VideoAppearanceEntry[]
  selectedTrackId?: string | null
  showOverlay?: boolean
  filterKnown?: boolean
  filterAnonymous?: boolean
  showLabels?: boolean
  onTrackSelect?: (trackId: string) => void
  onPlaybackTimeChange?: (timeSeconds: number) => void
  onError?: (message: string) => void
}

export const VideoOverlayPlayer = forwardRef<HTMLVideoElement, VideoOverlayPlayerProps>(
  ({
    src,
    fileName,
    className,
    displayWidth,
    displayHeight,
    rotationDegrees = 0,
    durationSeconds,
    frames,
    people: _people,
    selectedTrackId,
    showOverlay = true,
    filterKnown = true,
    filterAnonymous = true,
    showLabels = true,
    onTrackSelect: _onTrackSelect,
    onPlaybackTimeChange,
    onError,
  }, forwardedRef) => {
    const containerRef = useRef<HTMLDivElement>(null)
    const videoRef = useRef<HTMLVideoElement>(null)
    const setVideoRef = useCallback(
      (node: HTMLVideoElement | null) => {
        videoRef.current = node
        if (typeof forwardedRef === 'function') {
          forwardedRef(node)
        } else if (forwardedRef) {
          forwardedRef.current = node
        }
      },
      [forwardedRef],
    )
    const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isReady, setIsReady] = useState(false)
  const [isPlaying, setIsPlaying] = useState(false)
  const [currentTime, setCurrentTime] = useState(0)
  const rafHandleRef = useRef<number | null>(null)
  const frameCallbackRef = useRef<number | null>(null)
  const lastUiTimeRef = useRef(0)

  const rotation = rotationDegrees
  const sourceWidth = displayWidth
  const sourceHeight = displayHeight

  const resizeCanvas = useCallback(() => {
    const container = containerRef.current
    const canvas = canvasRef.current
    if (!container || !canvas) return
    const rect = container.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = Math.max(1, Math.floor(rect.width * dpr))
    canvas.height = Math.max(1, Math.floor(rect.height * dpr))
    canvas.style.width = `${rect.width}px`
    canvas.style.height = `${rect.height}px`
    const ctx = canvas.getContext('2d')
    if (ctx) {
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0)
      ctx.clearRect(0, 0, rect.width, rect.height)
    }
  }, [])

  const drawOverlays = useCallback(
    (timeSeconds: number) => {
      const canvas = canvasRef.current
      const container = containerRef.current
      if (!canvas || !container) return
      const ctx = canvas.getContext('2d')
      if (!ctx) return
      const rect = container.getBoundingClientRect()
      ctx.clearRect(0, 0, rect.width, rect.height)
      if (!showOverlay || !sourceWidth || !sourceHeight) return

      const videoRect = calculateContainedVideoRect(rect.width, rect.height, sourceWidth, sourceHeight, rotation)
      const visible = selectVisibleDetectionsAtTime(
        frames,
        timeSeconds,
        sourceWidth,
        sourceHeight,
        rotation,
        videoRect,
      )

      ctx.font = FONT
      for (const { detection, canvasBBox } of visible) {
        if (!passesFilter(detection, filterKnown, filterAnonymous)) continue
        const selected = selectedTrackId === detection.track_id
        const color = selected
          ? { background: '#f59e0b', text: 'white' }
          : stableTrackColor(detection.track_id)
        const lineWidth = selected ? BOX_LINE_SELECTED : BOX_LINE_NORMAL

        ctx.strokeStyle = color.background
        ctx.lineWidth = lineWidth
        ctx.strokeRect(canvasBBox.x, canvasBBox.y, canvasBBox.width, canvasBBox.height)

        if (!showLabels) continue

        const label = buildLabel(detection)
        const textMetrics = ctx.measureText(label)
        const labelHeight = 18
        const labelWidth = textMetrics.width + LABEL_PADDING * 2
        let labelX = canvasBBox.x
        let labelY = canvasBBox.y - labelHeight
        if (labelY < 0) {
          labelY = canvasBBox.y + labelHeight
        }
        if (labelX + labelWidth > rect.width) {
          labelX = rect.width - labelWidth
        }
        ctx.fillStyle = color.background
        ctx.fillRect(labelX, labelY, labelWidth, labelHeight)
        ctx.fillStyle = color.text
        ctx.fillText(label, labelX + LABEL_PADDING, labelY + 13)
      }
    },
    [frames, sourceWidth, sourceHeight, rotation, showOverlay, filterKnown, filterAnonymous, showLabels, selectedTrackId],
  )

  const scheduleFrame = useCallback(() => {
    const video = videoRef.current
    if (!video) return
    const update = () => {
      const t = video.currentTime
      drawOverlays(t)
      if (Date.now() - lastUiTimeRef.current > 100) {
        setCurrentTime(t)
        onPlaybackTimeChange?.(t)
        lastUiTimeRef.current = Date.now()
      }
    }
    if ('requestVideoFrameCallback' in video) {
      const cb = () => {
        update()
        frameCallbackRef.current = video.requestVideoFrameCallback(cb)
      }
      frameCallbackRef.current = video.requestVideoFrameCallback(cb)
    } else {
      const cb = () => {
        update()
        rafHandleRef.current = requestAnimationFrame(cb)
      }
      rafHandleRef.current = requestAnimationFrame(cb)
    }
  }, [drawOverlays, onPlaybackTimeChange])

  const cancelFrameLoop = useCallback(() => {
    const video = videoRef.current
    if (frameCallbackRef.current !== null && video && 'cancelVideoFrameCallback' in video) {
      video.cancelVideoFrameCallback(frameCallbackRef.current)
      frameCallbackRef.current = null
    }
    if (rafHandleRef.current !== null) {
      cancelAnimationFrame(rafHandleRef.current)
      rafHandleRef.current = null
    }
  }, [])

  useEffect(() => {
    resizeCanvas()
    const container = containerRef.current
    if (!container) return
    const observer = new ResizeObserver(resizeCanvas)
    observer.observe(container)
    return () => observer.disconnect()
  }, [resizeCanvas, src])

  useEffect(() => {
    drawOverlays(currentTime)
  }, [drawOverlays, currentTime])

  useEffect(() => {
    return () => cancelFrameLoop()
  }, [cancelFrameLoop])

  useEffect(() => {
    const video = videoRef.current
    if (!video) return
    const onPlay = () => {
      setIsPlaying(true)
      scheduleFrame()
    }
    const onPause = () => setIsPlaying(false)
    const onSeek = () => drawOverlays(video.currentTime)
    video.addEventListener('play', onPlay)
    video.addEventListener('pause', onPause)
    video.addEventListener('seeked', onSeek)
    return () => {
      video.removeEventListener('play', onPlay)
      video.removeEventListener('pause', onPause)
      video.removeEventListener('seeked', onSeek)
    }
  }, [drawOverlays, scheduleFrame])

  function handleLoadedData() {
    setIsReady(true)
    const video = videoRef.current
    if (video) {
      setCurrentTime(video.currentTime)
      drawOverlays(video.currentTime)
    }
  }

  function togglePlay() {
    const video = videoRef.current
    if (!video) return
    if (video.paused) {
      video.play().catch(() => undefined)
    } else {
      video.pause()
    }
  }

  if (!src) {
    return (
      <div
        className={cn(
          'flex min-h-[360px] flex-col items-center justify-center rounded-xl border border-navy-200 bg-navy-50 text-navy-400',
          className,
        )}
      >
        <div className="mb-3 rounded-full bg-navy-100 p-3 text-navy-300">
          <Film className="h-8 w-8" aria-hidden="true" />
        </div>
        <p className="font-medium text-navy-600">Başlamak için bir video yükleyin.</p>
        <p className="text-sm">MP4, MOV veya WEBM.</p>
      </div>
    )
  }

  return (
    <div ref={containerRef} className={cn('relative w-full overflow-hidden rounded-xl bg-black shadow-sm', className)}>
      <video
        ref={setVideoRef}
        src={src}
        controls
        preload="metadata"
        crossOrigin="anonymous"
        className="block h-full w-full object-contain"
        onLoadedData={handleLoadedData}
        onError={(e) => onError?.(formatVideoError(e.currentTarget.error))}
        aria-label={fileName ? `Video: ${fileName}` : 'İşlenen video'}
      />
      <canvas
        ref={canvasRef}
        className={cn('pointer-events-none absolute inset-0 z-10', isReady ? 'opacity-100' : 'opacity-0')}
        aria-hidden="true"
      />
      <div className="absolute right-3 top-3 z-20 flex items-center gap-2 rounded-lg bg-black/70 px-3 py-1.5 text-xs text-white backdrop-blur-sm">
        <button
          type="button"
          onClick={togglePlay}
          className="flex items-center justify-center rounded p-1 hover:bg-white/20 focus:outline-none focus-visible:ring-2 focus-visible:ring-white/50"
          aria-label={isPlaying ? 'Duraklat' : 'Oynat'}
        >
          {isPlaying ? <Pause className="h-4 w-4" /> : <Play className="h-4 w-4" />}
        </button>
        <span aria-live="polite">
          {formatMediaTimeSeconds(currentTime)} / {formatMediaTimeSeconds(durationSeconds)}
        </span>
      </div>
    </div>
  )
})
VideoOverlayPlayer.displayName = 'VideoOverlayPlayer'

function passesFilter(detection: OverlayDetection, filterKnown: boolean, filterAnonymous: boolean): boolean {
  const isKnown = detection.status === 'known'
  const isAnonymous = detection.status === 'anonymous' || detection.status === 'new_anonymous'
  if (isKnown) return filterKnown
  if (isAnonymous) return filterAnonymous
  return true
}

function buildLabel(detection: OverlayDetection): string {
  const display = detection.name || statusLabel(detection.status)
  return `${display} · ${Math.round(detection.confidence * 100)}%`
}

function statusLabel(status: OverlayDetection['status']): string {
  switch (status) {
    case 'known':
      return 'Bilinen'
    case 'anonymous':
      return 'Anonim'
    case 'new_anonymous':
      return 'Yeni anonim'
    default:
      return status
  }
}

function formatVideoError(error: MediaError | null): string {
  if (!error) return 'Video yüklenemedi.'
  switch (error.code) {
    case MediaError.MEDIA_ERR_ABORTED:
      return 'Video oynatma kullanıcı tarafından durduruldu.'
    case MediaError.MEDIA_ERR_NETWORK:
      return 'Video ağ hatası nedeniyle yüklenemedi.'
    case MediaError.MEDIA_ERR_DECODE:
      return 'Video formatı/codec’i tarayıcı tarafından desteklenmiyor.'
    case MediaError.MEDIA_ERR_SRC_NOT_SUPPORTED:
      return 'Bu video dosyası tarayıcıda desteklenmiyor.'
    default:
      return `Video yüklenemedi (kod: ${error.code}).`
  }
}
