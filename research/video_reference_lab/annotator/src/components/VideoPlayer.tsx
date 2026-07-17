import { useCallback, useEffect, useRef } from 'react';
import type { OverlayRecord } from '../api';

interface VideoPlayerProps {
  src: string;
  displayWidth: number;
  displayHeight: number;
  initialTimeSeconds?: number;
  overlayRecords: OverlayRecord[];
  selectedObservationId: string | null;
  onSelectObservation: (id: string | null) => void;
  onMediaTimeChanged: (mediaTimeSeconds: number) => void;
}

interface FitRect {
  x: number;
  y: number;
  width: number;
  height: number;
  scale: number;
}

function getVideoFitRect(
  containerWidth: number,
  containerHeight: number,
  videoWidth: number,
  videoHeight: number
): FitRect {
  if (videoWidth <= 0 || videoHeight <= 0) {
    return { x: 0, y: 0, width: containerWidth, height: containerHeight, scale: 1 };
  }
  const scale = Math.min(containerWidth / videoWidth, containerHeight / videoHeight);
  const width = videoWidth * scale;
  const height = videoHeight * scale;
  const x = (containerWidth - width) / 2;
  const y = (containerHeight - height) / 2;
  return { x, y, width, height, scale };
}

function colorForTracklet(rawTrackletId: string): string {
  let hash = 0;
  for (let i = 0; i < rawTrackletId.length; i++) {
    hash = rawTrackletId.charCodeAt(i) + ((hash << 5) - hash);
  }
  const hue = Math.abs(hash % 360);
  return `hsl(${hue} 70% 60%)`;
}

export default function VideoPlayer({
  src,
  displayWidth,
  displayHeight,
  initialTimeSeconds,
  overlayRecords,
  selectedObservationId,
  onSelectObservation,
  onMediaTimeChanged,
}: VideoPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video || initialTimeSeconds === undefined || !Number.isFinite(initialTimeSeconds)) return;
    if (video.readyState >= 1) {
      video.currentTime = initialTimeSeconds;
    } else {
      const handler = () => {
        video.currentTime = initialTimeSeconds ?? 0;
      };
      video.addEventListener('loadedmetadata', handler, { once: true });
      return () => video.removeEventListener('loadedmetadata', handler);
    }
  }, [initialTimeSeconds]);

  const draw = useCallback(() => {
    const container = containerRef.current;
    const canvas = canvasRef.current;
    if (!container || !canvas) return;

    const rect = container.getBoundingClientRect();
    const cssWidth = Math.round(rect.width);
    const cssHeight = Math.round(rect.height);

    if (canvas.width !== cssWidth || canvas.height !== cssHeight) {
      canvas.width = cssWidth;
      canvas.height = cssHeight;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const fit = getVideoFitRect(cssWidth, cssHeight, displayWidth, displayHeight);

    for (const record of overlayRecords) {
      const [x1, y1, x2, y2] = record.bbox_xyxy;
      const sx = fit.x + x1 * fit.scale;
      const sy = fit.y + y1 * fit.scale;
      const sw = (x2 - x1) * fit.scale;
      const sh = (y2 - y1) * fit.scale;

      const isSelected = record.observation_id === selectedObservationId;
      const color = colorForTracklet(record.raw_tracklet_id);

      ctx.strokeStyle = color;
      ctx.lineWidth = isSelected ? 3 : 1.5;
      ctx.strokeRect(sx, sy, sw, sh);

      if (isSelected) {
        ctx.fillStyle = color;
        ctx.globalAlpha = 0.15;
        ctx.fillRect(sx, sy, sw, sh);
        ctx.globalAlpha = 1;
      }

      const label = record.display_label ?? record.raw_tracklet_id;
      const text = `${label} ${(record.detector_score * 100).toFixed(0)}%`;
      ctx.font = '12px ui-sans-serif, system-ui, sans-serif';
      const textWidth = ctx.measureText(text).width;
      ctx.fillStyle = color;
      ctx.fillRect(sx, sy - 16, Math.max(textWidth + 8, sw), 16);
      ctx.fillStyle = '#000';
      ctx.fillText(text, sx + 4, sy - 4);
    }
  }, [displayWidth, displayHeight, overlayRecords, selectedObservationId]);

  useEffect(() => {
    draw();
  }, [draw]);

  useEffect(() => {
    const onResize = () => draw();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [draw]);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return;

    let handle: number | null = null;

    const schedule = () => {
      handle = video.requestVideoFrameCallback((_now: number, metadata: VideoFrameCallbackMetadata) => {
        onMediaTimeChanged(metadata.mediaTime);
        draw();
        schedule();
      });
    };

    if (typeof video.requestVideoFrameCallback === 'function') {
      schedule();
    } else {
      const onTimeUpdate = () => {
        onMediaTimeChanged(video.currentTime);
        draw();
      };
      video.addEventListener('timeupdate', onTimeUpdate);
      return () => {
        video.removeEventListener('timeupdate', onTimeUpdate);
      };
    }

    return () => {
      if (handle !== null) {
        video.cancelVideoFrameCallback(handle);
      }
    };
  }, [draw, onMediaTimeChanged]);

  const handleCanvasClick = (event: React.MouseEvent<HTMLCanvasElement>) => {
    const container = containerRef.current;
    if (!container) return;

    const containerRect = container.getBoundingClientRect();
    const mx = event.clientX - containerRect.left;
    const my = event.clientY - containerRect.top;

    const fit = getVideoFitRect(containerRect.width, containerRect.height, displayWidth, displayHeight);

    if (mx < fit.x || my < fit.y || mx > fit.x + fit.width || my > fit.y + fit.height) {
      onSelectObservation(null);
      return;
    }

    const vx = (mx - fit.x) / fit.scale;
    const vy = (my - fit.y) / fit.scale;

    let hit: OverlayRecord | null = null;
    for (let i = overlayRecords.length - 1; i >= 0; i--) {
      const r = overlayRecords[i];
      const [x1, y1, x2, y2] = r.bbox_xyxy;
      if (vx >= x1 && vy >= y1 && vx <= x2 && vy <= y2) {
        hit = r;
        break;
      }
    }

    onSelectObservation(hit ? hit.observation_id : null);
  };

  return (
    <div ref={containerRef} className="relative h-full w-full overflow-hidden bg-black">
      <video
        ref={videoRef}
        src={src}
        className="absolute inset-0 h-full w-full"
        style={{ objectFit: 'contain' }}
        controls={false}
        playsInline
        preload="auto"
        muted
      />
      <canvas
        ref={canvasRef}
        className="absolute inset-0 h-full w-full cursor-crosshair"
        onClick={handleCanvasClick}
      />
    </div>
  );
}
