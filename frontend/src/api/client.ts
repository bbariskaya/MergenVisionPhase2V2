import { ApiError, type ApiErrorResponse } from './types'

const API_BASE = '/api/v1'

function camelToSnake(key: string): string {
  return key.replace(/[A-Z]/g, (letter) => `_${letter.toLowerCase()}`)
}

function rekeyToSnake<T>(value: T): T {
  if (Array.isArray(value)) {
    return value.map((item) => rekeyToSnake(item)) as unknown as T
  }
  if (value !== null && typeof value === 'object') {
    const result: Record<string, unknown> = {}
    for (const [key, val] of Object.entries(value)) {
      result[camelToSnake(key)] = rekeyToSnake(val)
    }
    return result as unknown as T
  }
  return value
}

async function parseError(response: Response): Promise<ApiError> {
  let body: ApiErrorResponse | undefined
  try {
    body = (await response.json()) as ApiErrorResponse
  } catch {
    body = undefined
  }
  const message =
    typeof body?.detail === 'string'
      ? body.detail
      : `${response.status} ${response.statusText}`
  return new ApiError(message, response.status, body)
}

export async function apiFetch<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init.headers || {}),
    },
  })
  if (!response.ok) {
    throw await parseError(response)
  }
  if (response.status === 204) {
    return undefined as T
  }
  const json = (await response.json()) as T
  return rekeyToSnake(json)
}

export async function apiUpload<T>(
  path: string,
  formData: FormData,
  init: RequestInit = {},
): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    method: 'POST',
    body: formData,
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init.headers || {}),
    },
  })
  if (!response.ok) {
    throw await parseError(response)
  }
  const json = (await response.json()) as T
  return rekeyToSnake(json)
}
