export type FaceStatus = 'known' | 'anonymous' | 'new_anonymous'

export interface Face {
  id: string
  name: string | null
  status: FaceStatus
  metadata: Record<string, string>
  createdAt: string
  lastSeen: string
  sampleCount: number
}

export interface ProcessSummary {
  processId: string
  type: 'image' | 'video'
  timestamp: string
  faceCount: number
  personCount?: number
  durationMs: number
  status: 'completed' | 'failed' | 'processing' | 'pending'
}

export interface Detection {
  frame: number
  timestamp: number
  boundingBox: { x: number; y: number; width: number; height: number }
  confidence: number
}

export interface PersonTrack {
  faceId: string
  trackId: string
  status: FaceStatus
  name: string | null
  metadata: Record<string, string>
  firstSeen: number
  lastSeen: number
  totalDuration: number
  confidence: number
  appearances: { start: number; end: number; startFrame: number; endFrame: number }[]
  detections: Detection[]
}

export interface VideoJob {
  jobId: string
  processId: string
  status: 'completed' | 'processing' | 'pending' | 'failed'
  progress: number
  videoUrl: string
  duration: number
  width: number
  height: number
  totalFrames: number
  processedFrames: number
  samplingRate: string
  personCount: number
  persons: PersonTrack[]
}

export interface ImageResultFace {
  faceId: string
  status: FaceStatus
  name: string | null
  metadata: Record<string, string>
  boundingBox: { x: number; y: number; width: number; height: number }
  confidence: number
}

export interface ImageResult {
  processId: string
  faceCount: number
  faces: ImageResultFace[]
}

export const faces: Face[] = [
  { id: 'face_001', name: 'Ahmet Yılmaz', status: 'known', metadata: { department: 'Engineering' }, createdAt: '2026-07-10T10:00:00Z', lastSeen: '2026-07-16T08:30:00Z', sampleCount: 12 },
  { id: 'face_002', name: 'Zeynep Kaya', status: 'known', metadata: { department: 'Marketing' }, createdAt: '2026-07-11T12:00:00Z', lastSeen: '2026-07-16T09:15:00Z', sampleCount: 8 },
  { id: 'face_117', name: null, status: 'anonymous', metadata: {}, createdAt: '2026-07-15T14:20:00Z', lastSeen: '2026-07-16T10:00:00Z', sampleCount: 3 },
  { id: 'face_118', name: null, status: 'new_anonymous', metadata: {}, createdAt: '2026-07-16T09:50:00Z', lastSeen: '2026-07-16T09:50:00Z', sampleCount: 1 },
  { id: 'face_003', name: 'Murat Demir', status: 'known', metadata: { department: 'Security' }, createdAt: '2026-07-12T08:00:00Z', lastSeen: '2026-07-15T18:00:00Z', sampleCount: 15 },
]

export const processes: ProcessSummary[] = [
  { processId: 'proc_5d9b7c10', type: 'video', timestamp: '2026-07-16T10:00:00Z', faceCount: 2, personCount: 2, durationMs: 4200, status: 'completed' },
  { processId: 'proc_a8e2f19d', type: 'image', timestamp: '2026-07-16T09:45:00Z', faceCount: 3, durationMs: 850, status: 'completed' },
  { processId: 'proc_7c1b3a44', type: 'image', timestamp: '2026-07-16T09:30:00Z', faceCount: 1, durationMs: 620, status: 'completed' },
  { processId: 'proc_9f2e6b88', type: 'video', timestamp: '2026-07-16T09:00:00Z', faceCount: 0, personCount: 0, durationMs: 2100, status: 'completed' },
  { processId: 'proc_3d5a1c22', type: 'video', timestamp: '2026-07-16T08:50:00Z', faceCount: 4, personCount: 2, durationMs: 5600, status: 'processing' },
]

export const imageResult: ImageResult = {
  processId: 'proc_a8e2f19d',
  faceCount: 3,
  faces: [
    {
      faceId: 'face_001',
      status: 'known',
      name: 'Ahmet Yılmaz',
      metadata: { department: 'Engineering' },
      boundingBox: { x: 320, y: 210, width: 180, height: 180 },
      confidence: 0.94,
    },
    {
      faceId: 'face_002',
      status: 'known',
      name: 'Zeynep Kaya',
      metadata: { department: 'Marketing' },
      boundingBox: { x: 620, y: 250, width: 160, height: 160 },
      confidence: 0.91,
    },
    {
      faceId: 'face_117',
      status: 'anonymous',
      name: null,
      metadata: {},
      boundingBox: { x: 920, y: 300, width: 140, height: 140 },
      confidence: 0.76,
    },
  ],
}

const VIDEO_URL = 'https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/BigBuckBunny.mp4'

export const videoJob: VideoJob = {
  jobId: 'job_8f3c1a2e',
  processId: 'proc_5d9b7c10',
  status: 'completed',
  progress: 100,
  videoUrl: VIDEO_URL,
  duration: 42.5,
  width: 1920,
  height: 1080,
  totalFrames: 1275,
  processedFrames: 128,
  samplingRate: 'every_10th_frame',
  personCount: 2,
  persons: [
    {
      faceId: 'face_001',
      trackId: 'track_a1',
      status: 'known',
      name: 'Ahmet Yılmaz',
      metadata: { department: 'Engineering' },
      firstSeen: 1.2,
      lastSeen: 12.8,
      totalDuration: 11.6,
      confidence: 0.94,
      appearances: [{ start: 1.2, end: 12.8, startFrame: 36, endFrame: 384 }],
      detections: [
        { frame: 36, timestamp: 1.2, boundingBox: { x: 640, y: 220, width: 180, height: 180 }, confidence: 0.93 },
        { frame: 46, timestamp: 1.53, boundingBox: { x: 648, y: 224, width: 182, height: 181 }, confidence: 0.95 },
        { frame: 56, timestamp: 1.87, boundingBox: { x: 655, y: 230, width: 178, height: 179 }, confidence: 0.92 },
        { frame: 66, timestamp: 2.2, boundingBox: { x: 650, y: 228, width: 180, height: 180 }, confidence: 0.94 },
      ],
    },
    {
      faceId: 'face_117',
      trackId: 'track_b2',
      status: 'new_anonymous',
      name: null,
      metadata: {},
      firstSeen: 3.0,
      lastSeen: 9.4,
      totalDuration: 6.4,
      confidence: 0.81,
      appearances: [{ start: 3.0, end: 9.4, startFrame: 90, endFrame: 282 }],
      detections: [
        { frame: 90, timestamp: 3.0, boundingBox: { x: 1100, y: 300, width: 160, height: 160 }, confidence: 0.80 },
        { frame: 100, timestamp: 3.33, boundingBox: { x: 1102, y: 302, width: 162, height: 161 }, confidence: 0.82 },
        { frame: 110, timestamp: 3.67, boundingBox: { x: 1105, y: 305, width: 160, height: 160 }, confidence: 0.83 },
      ],
    },
  ],
}

export const analyticsData = {
  processVolume: [
    { date: 'Mon', image: 12, video: 4 },
    { date: 'Tue', image: 18, video: 6 },
    { date: 'Wed', image: 9, video: 3 },
    { date: 'Thu', image: 22, video: 8 },
    { date: 'Fri', image: 15, video: 5 },
    { date: 'Sat', image: 7, video: 2 },
    { date: 'Sun', image: 10, video: 4 },
  ],
  statusDistribution: [
    { name: 'Known', value: 58 },
    { name: 'Anonymous', value: 28 },
    { name: 'New Anonymous', value: 14 },
  ],
  confidenceTrend: [
    { time: '00:00', confidence: 0.82 },
    { time: '04:00', confidence: 0.85 },
    { time: '08:00', confidence: 0.91 },
    { time: '12:00', confidence: 0.89 },
    { time: '16:00', confidence: 0.93 },
    { time: '20:00', confidence: 0.88 },
  ],
  jobHealth: {
    completed: 86,
    failed: 5,
    processing: 4,
    pending: 2,
  },
}
