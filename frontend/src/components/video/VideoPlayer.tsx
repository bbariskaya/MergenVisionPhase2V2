import { cn } from '@/lib/utils'
import { Film } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'

export interface VideoPlayerProps {
  src: string | undefined
  fileName?: string
  className?: string
  onError?: (message: string) => void
}

export function VideoPlayer({ src, fileName, className, onError }: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [isReady, setIsReady] = useState(false)

  useEffect(() => {
    function resizeCanvas() {
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
    }

    resizeCanvas()
    const container = containerRef.current
    if (!container) return

    const observer = new ResizeObserver(resizeCanvas)
    observer.observe(container)

    return () => observer.disconnect()
  }, [src])

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
    <div
      ref={containerRef}
      className={cn(
        'relative w-full overflow-hidden rounded-xl bg-black shadow-sm',
        className,
      )}
    >
      <video
        src={src}
        controls
        preload="metadata"
        className="block h-full w-full object-contain"
        onLoadedData={() => setIsReady(true)}
        onError={(e) => onError?.(formatVideoError(e.currentTarget.error))}
        aria-label={fileName ? `Video: ${fileName}` : 'Yüklenen video'}
      />
      <canvas
        ref={canvasRef}
        className={cn(
          'pointer-events-none absolute inset-0 z-10',
          isReady ? 'opacity-100' : 'opacity-0',
        )}
        aria-hidden="true"
      />
    </div>
  )
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
