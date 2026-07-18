export const queryKeys = {
  health: {
    live: ['health', 'live'] as const,
    ready: ['health', 'ready'] as const,
  },
  face: (faceId: string) => ['face', faceId] as const,
  faceHistory: (faceId: string) => ['face', faceId, 'history'] as const,
  faceSamples: (faceId: string) => ['face', faceId, 'samples'] as const,
  faces: (params: { search?: string; isActive?: boolean | null; limit?: number; offset?: number }) =>
    ['faces', params] as const,
  process: (processId: string) => ['process', processId] as const,
  enrollmentStats: () => ['stats', 'enrollment'] as const,
  people: (params: { search?: string }) => ['people', params] as const,
  person: (personId: string) => ['person', personId] as const,
  bulkJob: (jobId: string) => ['bulk-job', jobId] as const,
  latestBulkJob: () => ['bulk-job', 'latest'] as const,
  video: (videoId: string) => ['video', videoId] as const,
  videoJob: (jobId: string) => ['video-job', jobId] as const,
  videoResult: (jobId: string) => ['video-result', jobId] as const,
  videoPeople: (jobId: string) => ['video-people', jobId] as const,
  videoAppearances: (jobId: string) => ['video-appearances', jobId] as const,
  videoTimelineFrames: (jobId: string, params: { start_pts_ns: number; end_pts_ns: number }) =>
    ['video-timeline-frames', jobId, params] as const,
}
