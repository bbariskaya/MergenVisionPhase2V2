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
  face_id: UUID
  status: string
  name: string
  metadata: Record<string, unknown> | null
}

export interface FaceDetail {
  face_id: UUID
  status: RecognitionStatus
  name: string | null
  metadata: Record<string, unknown> | null
  created_at: string
  updated_at: string
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
