import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import { apiFetch } from './client'
import { queryKeys } from './queryKeys'
import type {
  CreatePersonBatchRequest,
  CreatePersonRequest,
  PeopleBatchCreateResponse,
  PersonDetail,
  PersonListResponse,
  PersonSummary,
  UpdatePersonRequest,
} from './types'

export function usePeople(search?: string) {
  return useQuery({
    queryKey: queryKeys.people({ search }),
    queryFn: () => {
      const searchParams = new URLSearchParams()
      if (search && search.trim()) {
        searchParams.set('search', search.trim())
      }
      const query = searchParams.toString()
      return apiFetch<PersonListResponse>(`/people${query ? `?${query}` : ''}`)
    },
  })
}

export function usePerson(personId: string) {
  return useQuery({
    queryKey: queryKeys.person(personId),
    queryFn: () => apiFetch<PersonDetail>(`/people/${personId}`),
    enabled: personId.length > 0,
  })
}

export function useCreatePersonMutation() {
  const queryClient = useQueryClient()
  return useMutation<PersonSummary, Error, CreatePersonRequest>({
    mutationFn: async (body) =>
      apiFetch<PersonSummary>('/people', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.people({}) })
    },
  })
}

export function useCreatePeopleBatchMutation() {
  const queryClient = useQueryClient()
  return useMutation<PeopleBatchCreateResponse, Error, CreatePersonBatchRequest>({
    mutationFn: async (body) =>
      apiFetch<PeopleBatchCreateResponse>('/people/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.people({}) })
    },
  })
}

export function useUpdatePersonMutation() {
  const queryClient = useQueryClient()
  return useMutation<PersonSummary, Error, { personId: string; body: UpdatePersonRequest }>({
    mutationFn: async ({ personId, body }) =>
      apiFetch<PersonSummary>(`/people/${personId}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      }),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({ queryKey: queryKeys.person(variables.personId) })
      queryClient.invalidateQueries({ queryKey: queryKeys.people({}) })
    },
  })
}

export function useDeletePersonMutation() {
  const queryClient = useQueryClient()
  return useMutation<void, Error, string>({
    mutationFn: async (personId) =>
      apiFetch(`/people/${personId}`, { method: 'DELETE' }),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: queryKeys.people({}) })
    },
  })
}
