export type UUID = string

export interface BoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export type RecognitionStatus = 'known' | 'anonymous' | 'new_anonymous'

export interface RecognitionFace {
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  metadata: Record<string, unknown> | null
  bounding_box: BoundingBox
  confidence: number | null
}

export interface RecognizeResponse {
  process_id: UUID
  status: string
  face_count: number
  faces: RecognitionFace[]
}

export interface EnrollRequest {
  face_id: UUID
  name: string
  metadata?: Record<string, unknown>
}

export interface EnrollResponse {
  request_id: UUID
  process_id: UUID
  face_id: UUID
  person_id: UUID | null
  status: string
  name: string
  metadata: Record<string, unknown> | null
}

export interface FaceDetail {
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  person_id: UUID | null
  metadata: Record<string, unknown> | null
  created_at: string
  updated_at: string
}

export interface PersonSummary {
  person_id: UUID
  display_name: string
  is_active: boolean
  created_at: string
  updated_at: string
}

export interface PersonDetail {
  request_id: UUID
  person_id: UUID
  display_name: string
  is_active: boolean
  metadata: Record<string, unknown> | null
  face_count: number
  faces: IdentitySummary[]
  created_at: string
  updated_at: string
}

export interface PersonListResponse {
  request_id: UUID
  count: number
  people: PersonSummary[]
}

export interface CreatePersonRequest {
  display_name: string
  metadata?: Record<string, unknown>
}

export interface UpdatePersonRequest {
  display_name?: string
  metadata?: Record<string, unknown>
}

export interface CreatePersonBatchItem {
  display_name: string
  metadata?: Record<string, unknown>
}

export interface CreatePersonBatchRequest {
  people: CreatePersonBatchItem[]
}

export interface PeopleBatchCreateResponse {
  request_id: UUID
  count: number
  people: PersonSummary[]
}

export interface FaceHistoryEntry {
  process_id: UUID
  timestamp: string
  process_type?: string
  status?: string
  recognition_status?: string
  match_confidence?: number
}

export interface FaceHistoryResponse {
  face_id: UUID
  history: FaceHistoryEntry[]
}

export interface IdentitySummary {
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  metadata: Record<string, unknown> | null
  created_at: string | null
  updated_at: string | null
}

export interface IdentityListResponse {
  request_id: UUID
  count: number
  identities: IdentitySummary[]
}

export interface FaceSample {
  sample_id: UUID
  face_id: UUID
  state: string
  image_url: string | null
  created_at: string | null
  activated_at: string | null
}

export interface FaceSamplesResponse {
  request_id: UUID
  face_id: UUID
  count: number
  samples: FaceSample[]
}

export interface ProcessDetail {
  process_id: UUID
  process_type: string
  status: string
  face_count: number | null
  error_code: string | null
  details: { detections?: RecognitionFace[] } | null
  created_at: string
  completed_at: string | null
}

export interface HealthResponse {
  status: 'ok' | string
}

export interface ApiErrorDetail {
  loc: Array<string | number>
  msg: string
  type: string
}

export interface ApiErrorResponse {
  detail?: string | ApiErrorDetail[]
}

export class ApiError extends Error {
  constructor(
    message: string,
    public status: number,
    public body?: ApiErrorResponse,
  ) {
    super(message)
    this.name = 'ApiError'
  }
}

// --- Video schemas ---

export interface OverlayBoundingBox {
  x: number
  y: number
  width: number
  height: number
}

export interface OverlayDetection {
  track_id: UUID
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  bbox: OverlayBoundingBox
  confidence: number
  provenance: string
}

export interface OverlayFrame {
  frame_index: number
  pts_ns: number
  detections: OverlayDetection[]
}

export interface VideoRecognizeResponse {
  request_id: UUID
  process_id: UUID
  video_id: UUID
  job_id: UUID
  upload_session_id: UUID | null
  status: string
  status_url: string
  result_url: string
}

export interface VideoResponse {
  request_id: UUID
  video_id: UUID
  upload_session_id: UUID
  state: string
  content_sha256: string | null
  size_bytes: number | null
  container_format: string | null
  video_codec: string | null
  display_width: number | null
  display_height: number | null
  rotation_degrees: number
  duration_ns: number | null
  total_frames: number | null
  failure_code: string | null
}

export type VideoJobState =
  | 'pending'
  | 'processing'
  | 'cancelling'
  | 'completed'
  | 'failed'
  | 'cancelled'

export interface VideoJobResponse {
  request_id: UUID
  process_id: UUID
  video_id: UUID
  job_id: UUID
  state: VideoJobState
  stage: string
  progress_percent: number
  sampling_mode: string
  every_n_frames: number | null
  frames_per_second: number | null
  processed_frames: number
  sampled_frames: number
  detected_observations: number
  person_count: number
  cancellation_requested: boolean
  error_code: string | null
  created_at: string | null
  updated_at: string | null
  status_url: string
  result_url: string
}

export interface VideoJobResultResponse {
  request_id: UUID
  job_id: UUID
  state: VideoJobState
  result_available: boolean
  manifest_bucket: string | null
  manifest_key: string | null
  manifest_sha256: string | null
}

export interface VideoPersonSummary {
  track_id: UUID
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  current_status: RecognitionStatus | null
  current_name: string | null
  first_frame_index: number
  last_frame_index: number
  first_pts_ns: number
  last_pts_ns: number
  total_duration_ns: number
  detection_count: number
  appearance_count: number
  match_confidence: number
}

export interface VideoPeopleResponse {
  request_id: UUID
  job_id: UUID
  person_count: number
  people: VideoPersonSummary[]
}

export interface VideoAppearanceEntry {
  track_id: UUID
  face_id: UUID
  start_frame_index: number
  end_frame_index: number
  start_pts_ns: number
  end_pts_ns: number
  detection_count: number
}

export interface VideoAppearancesResponse {
  request_id: UUID
  job_id: UUID
  appearance_count: number
  appearances: VideoAppearanceEntry[]
}

export interface VideoTimelineRecord {
  track_id: UUID
  face_id: UUID
  start_frame_index: number
  end_frame_index: number
  start_pts_ns: number
  end_pts_ns: number
}

export interface VideoTimelineResponse {
  request_id: UUID
  job_id: UUID
  record_count: number
  records: VideoTimelineRecord[]
}

export interface VideoTimelineFramesResponse {
  request_id: UUID
  job_id: UUID
  start_pts_ns: number
  end_pts_ns: number
  record_count: number
  frames: OverlayFrame[]
}
