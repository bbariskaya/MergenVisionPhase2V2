import { useQuery } from '@tanstack/react-query'
import type { HealthResponse } from './types'

async function fetchHealth(): Promise<HealthResponse> {
  const response = await fetch('/health/live')
  if (!response.ok) throw new Error('health check failed')
  return response.json() as Promise<HealthResponse>
}

export function useHealth() {
  return useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    refetchInterval: 10_000,
  })
}
