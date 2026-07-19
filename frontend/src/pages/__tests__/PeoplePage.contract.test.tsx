import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { render, screen, waitFor } from '@testing-library/react'
import { MemoryRouter } from 'react-router'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import PeoplePage from '../PeoplePage'

describe('PeoplePage contract', () => {
  const fetchMock = vi.fn()

  beforeEach(() => {
    fetchMock.mockReset()
    globalThis.fetch = fetchMock as unknown as typeof fetch
  })

  function Wrapper({ children }: { children: React.ReactNode }) {
    const client = new QueryClient({
      defaultOptions: { queries: { retry: false, staleTime: 0 } },
    })
    return (
      <MemoryRouter>
        <QueryClientProvider client={client}>{children}</QueryClientProvider>
      </MemoryRouter>
    )
  }

  it('fetches identities from /api/v1/faces and links to /faces/:faceId', async () => {
    fetchMock.mockResolvedValueOnce({
      ok: true,
      status: 200,
      json: async () => ({
        request_id: 'req-1',
        count: 1,
        identities: [
          {
            face_id: 'face-id-1',
            status: 'known',
            name: 'Alice',
            metadata: null,
            created_at: null,
            updated_at: null,
          },
        ],
      }),
    } as Response)

    render(<PeoplePage />, { wrapper: Wrapper })

    await waitFor(() => expect(fetchMock).toHaveBeenCalled())

    const urls = fetchMock.mock.calls.map((call) => call[0] as string)
    expect(urls).toContain('/api/v1/faces')
    expect(urls).not.toContain('/api/v1/people')

    const link = await screen.findByRole('link', { name: /alice/i })
    expect(link).toHaveAttribute('href', '/faces/face-id-1')
  })
})
