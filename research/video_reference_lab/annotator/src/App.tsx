import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { api, type Config, type FrameInfo, type GroundTruthAnchor, type OverlayRecord, type Tracklet } from './api';
import VideoPlayer from './components/VideoPlayer';

type LoadingState = 'loading' | 'error' | 'ready';

function formatTime(ns: number): string {
  const totalSeconds = ns / 1_000_000_000;
  const minutes = Math.floor(totalSeconds / 60);
  const seconds = Math.floor(totalSeconds % 60);
  const ms = Math.floor((totalSeconds % 1) * 1000);
  return `${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}.${ms.toString().padStart(3, '0')}`;
}

function findNearestProcessedFrame(frames: FrameInfo[], mediaTimeSeconds: number): number {
  const targetPts = mediaTimeSeconds * 1_000_000_000;
  let best: FrameInfo | null = null;
  let bestDist = Infinity;
  for (const f of frames) {
    if (!f.processed) continue;
    const dist = Math.abs(f.pts_ns - targetPts);
    if (dist < bestDist) {
      bestDist = dist;
      best = f;
    }
  }
  return best?.frame_index ?? -1;
}

function generateId(): string {
  if (typeof crypto !== 'undefined' && 'randomUUID' in crypto) {
    return crypto.randomUUID();
  }
  return `${Date.now()}-${Math.random().toString(36).slice(2)}`;
}

export default function App() {
  const getVideo = () => document.querySelector('video') as HTMLVideoElement | null;

  const [state, setState] = useState<LoadingState>('loading');
  const [error, setError] = useState<string | null>(null);
  const [config, setConfig] = useState<Config | null>(null);
  const [frames, setFrames] = useState<FrameInfo[]>([]);
  const [overlayByFrame, setOverlayByFrame] = useState<Record<string, OverlayRecord[]>>({});
  const [tracklets, setTracklets] = useState<Tracklet[]>([]);
  const [anchors, setAnchors] = useState<GroundTruthAnchor[]>([]);

  const [mediaTimeSeconds, setMediaTimeSeconds] = useState(0);
  const [currentFrameIndex, setCurrentFrameIndex] = useState(-1);
  const [initialTimeSeconds, setInitialTimeSeconds] = useState<number | undefined>(undefined);
  const [selectedObservationId, setSelectedObservationId] = useState<string | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);

  const [label, setLabel] = useState('');
  const [split, setSplit] = useState<'calibration' | 'holdout'>('calibration');

  const [saveStatus, setSaveStatus] = useState<string | null>(null);
  const pendingSeekRef = useRef<{ timeSeconds: number; frameIndex: number } | null>(null);

  useEffect(() => {
    Promise.all([
      api.getConfig(),
      api.getFrames(),
      api.getOverlay(),
      api.getTracklets(),
      api.getGroundTruth(),
    ])
      .then(([cfg, frms, ovl, trks, gt]) => {
        setConfig(cfg);
        setFrames(frms.frames);
        setOverlayByFrame(ovl.by_frame);
        setTracklets(trks);
        setAnchors(gt.anchors ?? []);
        const processed = frms.frames.filter((f) => f.processed);
        const firstWithOverlay = processed.find(
          (f) => (ovl.by_frame[f.frame_index.toString()] ?? []).length > 0
        );
        const firstFrame = firstWithOverlay ?? processed[0];
        if (firstFrame) {
          const timeSeconds = firstFrame.pts_ns / 1_000_000_000;
          pendingSeekRef.current = { timeSeconds, frameIndex: firstFrame.frame_index };
          setCurrentFrameIndex(firstFrame.frame_index);
          setMediaTimeSeconds(timeSeconds);
          setInitialTimeSeconds(timeSeconds);
        }
        setState('ready');
      })
      .catch((err) => {
        setError(err instanceof Error ? err.message : String(err));
        setState('error');
      });
  }, []);

  const processedFrames = useMemo(() => frames.filter((f) => f.processed), [frames]);

  const currentOverlay = useMemo(() => {
    return overlayByFrame[currentFrameIndex.toString()] ?? [];
  }, [overlayByFrame, currentFrameIndex]);

  const selectedRecord = useMemo(() => {
    return currentOverlay.find((r) => r.observation_id === selectedObservationId) ?? null;
  }, [currentOverlay, selectedObservationId]);

  const existingAnchor = useMemo(() => {
    if (!selectedObservationId) return null;
    return anchors.find((a) => a.observation_id === selectedObservationId) ?? null;
  }, [anchors, selectedObservationId]);

  useEffect(() => {
    if (existingAnchor) {
      setLabel(existingAnchor.label);
      setSplit(existingAnchor.split);
    } else if (selectedRecord) {
      setLabel(selectedRecord.display_label ?? '');
      setSplit('calibration');
    } else {
      setLabel('');
      setSplit('calibration');
    }
  }, [existingAnchor, selectedRecord]);

  useEffect(() => {
    const video = getVideo();
    if (!video) return;
    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    video.addEventListener('play', onPlay);
    video.addEventListener('pause', onPause);
    return () => {
      video.removeEventListener('play', onPlay);
      video.removeEventListener('pause', onPause);
    };
  }, []);

  const onMediaTimeChanged = useCallback(
    (timeSeconds: number) => {
      const pending = pendingSeekRef.current;
      if (pending) {
        if (Math.abs(timeSeconds - pending.timeSeconds) < 0.02) {
          pendingSeekRef.current = null;
          setMediaTimeSeconds(timeSeconds);
        }
        return;
      }
      setMediaTimeSeconds(timeSeconds);
      setCurrentFrameIndex(findNearestProcessedFrame(frames, timeSeconds));
    },
    [frames]
  );

  const seekToFrame = useCallback(
    (frameIndex: number) => {
      const frame = frames.find((f) => f.frame_index === frameIndex);
      const video = getVideo();
      if (!frame || !video || !Number.isFinite(frame.pts_ns)) return;
      const timeSeconds = frame.pts_ns / 1_000_000_000;
      pendingSeekRef.current = { timeSeconds, frameIndex: frame.frame_index };
      video.currentTime = timeSeconds;
      setMediaTimeSeconds(timeSeconds);
      setCurrentFrameIndex(frame.frame_index);
      setSelectedObservationId(null);
    },
    [frames]
  );

  const togglePlayPause = useCallback(() => {
    const video = getVideo();
    if (!video) return;
    if (video.paused) {
      void video.play();
      setIsPlaying(true);
    } else {
      video.pause();
      setIsPlaying(false);
    }
  }, []);

  const jumpProcessed = useCallback(
    (direction: -1 | 1) => {
      if (processedFrames.length === 0) return;
      const currentIdx = processedFrames.findIndex((f) => f.frame_index === currentFrameIndex);
      let nextIdx: number;
      if (currentIdx === -1) {
        nextIdx = direction === 1 ? 0 : processedFrames.length - 1;
      } else {
        nextIdx = Math.min(Math.max(currentIdx + direction, 0), processedFrames.length - 1);
      }
      seekToFrame(processedFrames[nextIdx].frame_index);
    },
    [processedFrames, currentFrameIndex, seekToFrame]
  );

  const jumpToTracklet = useCallback(
    (tracklet: Tracklet) => {
      const framesInRange = processedFrames.filter(
        (f) => f.frame_index >= tracklet.first_frame_index && f.frame_index <= tracklet.last_frame_index
      );
      if (framesInRange.length === 0) return;
      seekToFrame(framesInRange[0].frame_index);
    },
    [processedFrames, seekToFrame]
  );

  const saveAnchor = useCallback(async () => {
    if (!selectedObservationId || !selectedRecord) {
      setSaveStatus('No face selected');
      return;
    }
    if (!label.trim()) {
      setSaveStatus('Label required');
      return;
    }

    const newAnchor: GroundTruthAnchor = {
      anchor_id: existingAnchor?.anchor_id ?? generateId(),
      label: label.trim(),
      split,
      frame_index: currentFrameIndex,
      observation_id: selectedObservationId,
    };

    const nextAnchors = existingAnchor
      ? anchors.map((a) => (a.anchor_id === existingAnchor.anchor_id ? newAnchor : a))
      : [...anchors, newAnchor];

    setSaveStatus('Saving…');
    try {
      await api.saveGroundTruth(nextAnchors);
      setAnchors(nextAnchors);
      setSaveStatus(`Saved ${nextAnchors.length} anchor${nextAnchors.length === 1 ? '' : 's'}`);
    } catch (err) {
      setSaveStatus(err instanceof Error ? err.message : 'Save failed');
    }
  }, [selectedObservationId, selectedRecord, label, split, existingAnchor, anchors, currentFrameIndex]);

  const deleteAnchor = useCallback(
    async (anchorId: string) => {
      const nextAnchors = anchors.filter((a) => a.anchor_id !== anchorId);
      try {
        await api.saveGroundTruth(nextAnchors);
        setAnchors(nextAnchors);
      } catch (err) {
        setSaveStatus(err instanceof Error ? err.message : 'Delete failed');
      }
    },
    [anchors]
  );

  useEffect(() => {
    const handler = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        if (event.key === 's' || event.key === 'S') return;
      }

      switch (event.key) {
        case ' ':
          event.preventDefault();
          togglePlayPause();
          break;
        case 'ArrowLeft':
          event.preventDefault();
          jumpProcessed(-1);
          break;
        case 'ArrowRight':
          event.preventDefault();
          jumpProcessed(1);
          break;
        case 's':
        case 'S':
          event.preventDefault();
          void saveAnchor();
          break;
      }
    };

    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [jumpProcessed, saveAnchor, togglePlayPause]);

  const durationSeconds = config ? config.duration_ns / 1_000_000_000 : 0;

  if (state === 'loading') {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-900 text-neutral-100">
        Loading annotator…
      </div>
    );
  }

  if (state === 'error' || !config) {
    return (
      <div className="flex h-screen items-center justify-center bg-neutral-900 px-6 text-red-400">
        Error: {error ?? 'could not load config'}
      </div>
    );
  }

  return (
    <div className="flex h-screen overflow-hidden bg-neutral-900 text-neutral-100">
      <div className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 overflow-hidden">
          <VideoPlayer
            src="/api/video"
            displayWidth={config.display_width}
            displayHeight={config.display_height}
            initialTimeSeconds={initialTimeSeconds}
            overlayRecords={currentOverlay}
            selectedObservationId={selectedObservationId}
            onSelectObservation={setSelectedObservationId}
            onMediaTimeChanged={onMediaTimeChanged}
          />
        </div>

        <div className="flex h-28 shrink-0 flex-col justify-center gap-3 border-t border-neutral-700 bg-neutral-800 px-4">
          <div className="flex items-center justify-between text-sm text-neutral-400">
            <span>
              Frame {currentFrameIndex >= 0 ? currentFrameIndex : '-'} / {config.decoded_frame_count - 1}
            </span>
            <span>
              {formatTime(mediaTimeSeconds * 1_000_000_000)} / {formatTime(config.duration_ns)}
            </span>
            <span>{processedFrames.length} processed</span>
          </div>

          <div className="flex items-center gap-3">
            <button
              onClick={togglePlayPause}
              className="rounded bg-neutral-700 px-3 py-2 text-sm hover:bg-neutral-600 focus:outline-none focus:ring-2 focus:ring-neutral-500"
            >
              {isPlaying ? 'Pause' : 'Play'}
            </button>
            <button
              onClick={() => jumpProcessed(-1)}
              className="rounded bg-neutral-700 px-3 py-2 text-sm hover:bg-neutral-600 focus:outline-none focus:ring-2 focus:ring-neutral-500"
            >
              Prev
            </button>
            <button
              onClick={() => jumpProcessed(1)}
              className="rounded bg-neutral-700 px-3 py-2 text-sm hover:bg-neutral-600 focus:outline-none focus:ring-2 focus:ring-neutral-500"
            >
              Next
            </button>

            <input
              type="range"
              min={0}
              max={durationSeconds || 0}
              step={0.001}
              value={mediaTimeSeconds}
              onChange={(e) => {
                const video = getVideo();
                if (!video) return;
                video.currentTime = Number(e.target.value);
              }}
              className="flex-1 cursor-pointer"
            />
          </div>
        </div>
      </div>

      <div className="flex w-80 shrink-0 flex-col border-l border-neutral-700 bg-neutral-800">
        <div className="border-b border-neutral-700 p-4">
          <h2 className="mb-3 text-sm font-semibold uppercase tracking-wide text-neutral-400">Selected Face</h2>
          {!selectedRecord ? (
            <p className="text-sm text-neutral-500">Click a bounding box to label a face.</p>
          ) : (
            <div className="space-y-3">
              <div className="text-xs text-neutral-400">
                <div>Observation: {selectedRecord.observation_id}</div>
                <div>Frame: {currentFrameIndex}</div>
                <div>Tracklet: {selectedRecord.raw_tracklet_id}</div>
              </div>

              <div>
                <label className="mb-1 block text-xs text-neutral-400">Label</label>
                <input
                  type="text"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="e.g. alice"
                  className="w-full rounded border border-neutral-600 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 placeholder-neutral-600 focus:border-neutral-400 focus:outline-none"
                />
              </div>

              <div>
                <label className="mb-1 block text-xs text-neutral-400">Split</label>
                <select
                  value={split}
                  onChange={(e) => setSplit(e.target.value as 'calibration' | 'holdout')}
                  className="w-full rounded border border-neutral-600 bg-neutral-900 px-2 py-1.5 text-sm text-neutral-100 focus:border-neutral-400 focus:outline-none"
                >
                  <option value="calibration">calibration</option>
                  <option value="holdout">holdout</option>
                </select>
              </div>

              <button
                onClick={() => void saveAnchor()}
                className="w-full rounded bg-blue-600 px-3 py-2 text-sm font-medium hover:bg-blue-500 focus:outline-none focus:ring-2 focus:ring-blue-400"
              >
                Save (S)
              </button>

              {saveStatus && <div className="text-xs text-neutral-400">{saveStatus}</div>}
            </div>
          )}
        </div>

        <div className="flex-1 overflow-y-auto border-b border-neutral-700 p-4">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">Tracklets</h2>
          <ul className="space-y-1">
            {tracklets.map((t) => (
              <li key={t.raw_tracklet_id}>
                <button
                  onClick={() => jumpToTracklet(t)}
                  className="w-full rounded px-2 py-1.5 text-left text-sm hover:bg-neutral-700 focus:outline-none focus:ring-2 focus:ring-neutral-500"
                >
                  <div className="flex items-center justify-between">
                    <span className="truncate font-medium">{t.raw_tracklet_id}</span>
                    <span className="text-xs text-neutral-500">{t.observation_count}</span>
                  </div>
                  <div className="text-xs text-neutral-500">
                    {t.first_frame_index}–{t.last_frame_index} · {t.state}
                  </div>
                </button>
              </li>
            ))}
          </ul>
        </div>

        <div className="h-1/3 overflow-y-auto p-4">
          <h2 className="mb-2 text-sm font-semibold uppercase tracking-wide text-neutral-400">
            Anchors ({anchors.length})
          </h2>
          <ul className="space-y-1">
            {anchors.map((a) => (
              <li
                key={a.anchor_id}
                className="flex items-start justify-between rounded px-2 py-1.5 text-sm hover:bg-neutral-700"
              >
                <button
                  onClick={() => {
                    setSelectedObservationId(a.observation_id);
                    seekToFrame(a.frame_index);
                  }}
                  className="min-w-0 text-left"
                >
                  <div className="truncate font-medium">{a.label}</div>
                  <div className="text-xs text-neutral-500">
                    {a.split} · frame {a.frame_index}
                  </div>
                </button>
                <button
                  onClick={() => void deleteAnchor(a.anchor_id)}
                  className="ml-2 shrink-0 text-xs text-red-400 hover:text-red-300"
                >
                  del
                </button>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
