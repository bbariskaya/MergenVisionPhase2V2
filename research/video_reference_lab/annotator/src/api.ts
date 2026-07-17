export interface Config {
  video_path: string;
  display_width: number;
  display_height: number;
  duration_ns: number;
  decoded_frame_count: number;
  sampled_frame_count: number;
  processed_frame_count: number;
  strategy: string;
  run_dir: string;
  overlay_path: string;
  gt_path: string;
}

export interface FrameInfo {
  frame_index: number;
  pts_ns: number;
  sampled: boolean;
  processed: boolean;
}

export interface FramesResponse {
  count: number;
  frames: FrameInfo[];
}

export interface OverlayRecord {
  observation_id: string;
  pts_ns: number;
  bbox_xyxy: [number, number, number, number];
  raw_tracklet_id: string;
  canonical_track_id: string;
  display_label: string | null;
  detector_score: number;
  quality_score: number;
  recognition_eligible: boolean;
  rejection_reasons: string[];
}

export interface OverlayResponse {
  by_frame: Record<string, OverlayRecord[]>;
}

export interface Tracklet {
  raw_tracklet_id: string;
  first_frame_index: number;
  last_frame_index: number;
  observation_count: number;
  state: string;
}

export interface GroundTruthAnchor {
  anchor_id: string;
  label: string;
  split: 'calibration' | 'holdout';
  frame_index: number;
  observation_id: string;
}

export interface GroundTruth {
  anchors: GroundTruthAnchor[];
}

async function apiGet<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) {
    throw new Error(`GET ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

async function apiPost<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) {
    throw new Error(`POST ${path} failed: ${res.status} ${res.statusText}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  getConfig: () => apiGet<Config>('/api/config'),
  getFrames: () => apiGet<FramesResponse>('/api/frames'),
  getOverlay: () => apiGet<OverlayResponse>('/api/overlay'),
  getTracklets: () => apiGet<Tracklet[]>('/api/tracklets'),
  getGroundTruth: () => apiGet<GroundTruth>('/api/gt'),
  saveGroundTruth: (anchors: GroundTruthAnchor[]) =>
    apiPost<{ saved: boolean; anchor_count: number }>('/api/gt', { anchors }),
};
