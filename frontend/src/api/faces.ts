import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch, apiUpload } from './client'
import { queryKeys } from './queryKeys'
import type {
  EnrollRequest,
  EnrollResponse,
  FaceDetail,
  FaceHistoryResponse,
  FaceSample,
  FaceSamplesResponse,
  IdentityListResponse,
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
      return apiFetch<EnrollResponse>(`/faces/${face_id}/enroll`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, metadata }),
      })
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.face(variables.face_id) })
      queryClient.invalidateQueries({ queryKey: queryKeys.faceHistory(variables.face_id) })
      queryClient.invalidateQueries({ queryKey: queryKeys.faces({}) })
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
      queryClient.invalidateQueries({ queryKey: queryKeys.faces({}) })
    },
  })
}

export interface AddFaceSampleVariables {
  faceId: string
  image: File
}

export function useAddFaceSampleMutation() {
  const queryClient = useQueryClient()
  return useMutation<FaceSample, Error, AddFaceSampleVariables>({
    mutationFn: async ({ faceId, image }) => {
      const formData = new FormData()
      formData.append('image', image)
      return apiUpload<FaceSample>(`/faces/${faceId}/samples`, formData)
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.faceSamples(variables.faceId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.face(variables.faceId) })
    },
  })
}

export interface DeleteFaceSampleVariables {
  faceId: string
  sampleId: string
}

export function useDeleteFaceSampleMutation() {
  const queryClient = useQueryClient()
  return useMutation<void, Error, DeleteFaceSampleVariables>({
    mutationFn: async ({ faceId, sampleId }) => {
      return apiFetch(`/faces/${faceId}/samples/${sampleId}`, { method: 'DELETE' })
    },
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.faceSamples(variables.faceId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.face(variables.faceId) })
    },
  })
}

export function useFaces(params: { search?: string } = {}) {
  return useQuery({
    queryKey: queryKeys.faces(params),
    queryFn: () => {
      const searchParams = new URLSearchParams()
      if (params.search && params.search.trim()) {
        searchParams.set('search', params.search.trim())
      }
      const query = searchParams.toString()
      return apiFetch<IdentityListResponse>(`/faces${query ? `?${query}` : ''}`)
    },
  })
}

export function useFaceSamples(faceId: string) {
  return useQuery({
    queryKey: queryKeys.faceSamples(faceId),
    queryFn: () => apiFetch<FaceSamplesResponse>(`/faces/${faceId}/samples`),
    enabled: faceId.length > 0,
  })
}
