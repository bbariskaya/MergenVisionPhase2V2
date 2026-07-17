import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch, apiUpload } from './client'
import { queryKeys } from './queryKeys'
import type {
  EnrollRequest,
  EnrollResponse,
  FaceDetail,
  FaceHistoryResponse,
  RecognizeResponse,
} from './types'

export interface RecognizeVariables {
  image: File
}

export function useRecognizeMutation() {
  return useMutation<RecognizeResponse, Error, RecognizeVariables>({
    mutationFn: async ({ image }) => {
      const formData = new FormData()
      formData.append('image', image)
      return apiUpload<RecognizeResponse>('/faces/recognize', formData)
    },
  })
}

export function useEnrollMutation() {
  const queryClient = useQueryClient()
  return useMutation<EnrollResponse, Error, EnrollRequest>({
    mutationFn: async ({ face_id, name, metadata }) => {
      return apiFetch<EnrollResponse>('/faces/enroll', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ face_id, name, metadata }),
      })
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.face(variables.face_id) })
      queryClient.invalidateQueries({ queryKey: queryKeys.faceHistory(variables.face_id) })
    },
  })
}

export function useFace(faceId: string) {
  return useQuery({
    queryKey: queryKeys.face(faceId),
    queryFn: () => apiFetch<FaceDetail>(`/faces/${faceId}`),
    enabled: faceId.length > 0,
  })
}

export function useFaceHistory(faceId: string) {
  return useQuery({
    queryKey: queryKeys.faceHistory(faceId),
    queryFn: () => apiFetch<FaceHistoryResponse>(`/faces/${faceId}/history`),
    enabled: faceId.length > 0,
  })
}

export function useDeleteFaceMutation() {
  const queryClient = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: (faceId) => apiFetch(`/faces/${faceId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['face'] })
    },
  })
}
