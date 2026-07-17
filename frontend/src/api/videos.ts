import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch, apiUpload } from './client'
import { queryKeys } from './queryKeys'
import type {
  VideoAppearancesResponse,
  VideoJobResponse,
  VideoJobResultResponse,
  VideoPeopleResponse,
  VideoRecognizeResponse,
  VideoResponse,
  VideoTimelineFramesResponse,
  VideoTimelineResponse,
} from './types'

export interface UploadVideoVariables {
  file: File
  idempotencyKey: string
  samplingMode?: 'every_frame' | 'every_n_frames' | 'frames_per_second'
  everyNFrames?: number
  framesPerSecond?: number
}

export function useUploadVideoMutation() {
  const queryClient = useQueryClient()
  return useMutation<VideoRecognizeResponse, Error, UploadVideoVariables>({
    mutationFn: async ({ file, idempotencyKey, samplingMode, everyNFrames, framesPerSecond }) => {
      const formData = new FormData()
      formData.append('video', file)
      formData.append('samplingMode', samplingMode ?? 'every_frame')
      if (everyNFrames !== undefined) {
        formData.append('everyNFrames', String(everyNFrames))
      }
      if (framesPerSecond !== undefined) {
        formData.append('framesPerSecond', String(framesPerSecond))
      }
      return apiUpload<VideoRecognizeResponse>('/videos/recognize', formData, {
        headers: { 'Idempotency-Key': idempotencyKey },
      })
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['videos'] })
    },
  })
}

export function useVideo(videoId: string) {
  return useQuery({
    queryKey: queryKeys.video(videoId),
    queryFn: () => apiFetch<VideoResponse>(`/videos/${videoId}`),
    enabled: videoId.length > 0,
  })
}

export function useVideoJob(jobId: string) {
  return useQuery({
    queryKey: queryKeys.videoJob(jobId),
    queryFn: () => apiFetch<VideoJobResponse>(`/videos/jobs/${jobId}`),
    enabled: jobId.length > 0,
    refetchInterval: (query) => {
      const data = query.state.data
      if (!data) return 2000
      if (['pending', 'processing', 'cancelling'].includes(data.state)) {
        return 2000
      }
      return false
    },
  })
}

export function useVideoJobResult(jobId: string) {
  return useQuery({
    queryKey: queryKeys.videoResult(jobId),
    queryFn: () => apiFetch<VideoJobResultResponse>(`/videos/jobs/${jobId}/result`),
    enabled: jobId.length > 0,
    refetchInterval: (query) => {
      const data = query.state.data
      if (data?.result_available) return false
      return 1000
    },
  })
}

export function useCancelVideoJobMutation() {
  const queryClient = useQueryClient()
  return useMutation<VideoJobResponse, Error, string>({
    mutationFn: async (jobId) => {
      return apiFetch<VideoJobResponse>(`/videos/jobs/${jobId}`, { method: 'DELETE' })
    },
    onSuccess: (_, jobId) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.videoJob(jobId) })
    },
  })
}

export interface RetryVideoVariables {
  jobId: string
  idempotencyKey: string
}

export function useRetryVideoMutation() {
  return useMutation<VideoRecognizeResponse, Error, RetryVideoVariables>({
    mutationFn: async ({ jobId, idempotencyKey }) => {
      return apiFetch<VideoRecognizeResponse>(`/videos/jobs/${jobId}/retry`, {
        method: 'POST',
        headers: { 'Idempotency-Key': idempotencyKey },
      })
    },
  })
}

export function useVideoPeople(jobId: string) {
  return useQuery({
    queryKey: queryKeys.videoPeople(jobId),
    queryFn: () => apiFetch<VideoPeopleResponse>(`/videos/jobs/${jobId}/people`),
    enabled: jobId.length > 0,
  })
}

export function useVideoAppearances(jobId: string) {
  return useQuery({
    queryKey: queryKeys.videoAppearances(jobId),
    queryFn: () => apiFetch<VideoAppearancesResponse>(`/videos/jobs/${jobId}/appearances`),
    enabled: jobId.length > 0,
  })
}

export function useVideoTimeline(jobId: string) {
  return useQuery({
    queryKey: ['video-timeline', jobId],
    queryFn: () => apiFetch<VideoTimelineResponse>(`/videos/jobs/${jobId}/timeline`),
    enabled: jobId.length > 0,
  })
}

export function useVideoOverlayFrames(
  jobId: string,
  startPtsNs: number | undefined,
  endPtsNs: number | undefined,
  enabled: boolean,
) {
  return useQuery<VideoTimelineFramesResponse, Error>({
    queryKey: queryKeys.videoTimelineFrames(jobId, {
      start_pts_ns: startPtsNs ?? 0,
      end_pts_ns: endPtsNs ?? 0,
    }),
    queryFn: () => {
      const params = new URLSearchParams()
      const start = startPtsNs ?? 0
      params.set('start_pts_ns', String(start))
      if (endPtsNs !== undefined) {
        params.set('end_pts_ns', String(endPtsNs))
      }
      return apiFetch<VideoTimelineFramesResponse>(`/videos/jobs/${jobId}/timeline/frames?${params}`)
    },
    enabled: jobId.length > 0 && enabled && startPtsNs !== undefined,
  })
}

export function buildPlaybackUrl(videoId: string): string {
  return `/api/v1/videos/${videoId}/playback`
}
