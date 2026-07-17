import { Badge } from '@/components/ui/Badge'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Skeleton } from '@/components/ui/Skeleton'
import { cn } from '@/lib/utils'
import { formatDurationNs, stableTrackColor } from '@/lib/video'
import { Clock, Eye, UserPlus, Users } from 'lucide-react'
import { Link } from 'react-router'
import type { VideoPersonSummary } from '@/api/types'

export interface TrackListPanelProps {
  people: VideoPersonSummary[]
  selectedTrackId?: string | null
  filterKnown: boolean
  filterAnonymous: boolean
  isLoading?: boolean
  onSelectTrack: (trackId: string) => void
  onSeekToTrack: (trackId: string, ptsNs: number) => void
}

export function TrackListPanel({
  people,
  selectedTrackId,
  filterKnown,
  filterAnonymous,
  isLoading,
  onSelectTrack,
  onSeekToTrack,
}: TrackListPanelProps) {
  const filtered = people.filter((p) => {
    if (p.status === 'known' && !filterKnown) return false
    if ((p.status === 'anonymous' || p.status === 'new_anonymous') && !filterAnonymous) return false
    return true
  })

  if (isLoading) {
    return (
      <Card className="h-full">
        <CardContent className="space-y-4 p-4">
          {Array.from({ length: 5 }).map((_, i) => (
            <Skeleton key={i} className="h-20 w-full" />
          ))}
        </CardContent>
      </Card>
    )
  }

  if (filtered.length === 0) {
    return (
      <Card className="h-full">
        <CardContent className="p-4">
          <EmptyState
            icon={Users}
            title="Kimlik bulunamadı"
            description="Seçili filtrelere uygun kişi yok veya video henüz işlenmedi."
          />
        </CardContent>
      </Card>
    )
  }

  return (
    <Card className="h-full">
      <CardContent className="space-y-3 p-4">
        <h2 className="text-sm font-semibold text-navy-900">Kişiler</h2>
        <ul className="space-y-2" role="listbox" aria-label="Tespit edilen kişiler">
          {filtered.map((person) => {
            const selected = selectedTrackId === person.track_id
            const color = stableTrackColor(person.track_id)
            return (
              <li key={person.track_id}>
                <div
                  role="button"
                  tabIndex={0}
                  aria-selected={selected}
                  onClick={() => onSelectTrack(person.track_id)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      e.preventDefault()
                      onSelectTrack(person.track_id)
                    }
                  }}
                  className={cn(
                    'w-full rounded-lg border p-3 text-left transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                    selected ? 'border-primary bg-primary-50 ring-1 ring-primary' : 'border-navy-200 bg-white hover:bg-navy-50',
                  )}
                >
                  <div className="flex items-start gap-3">
                    <div
                      className="mt-0.5 h-3 w-3 shrink-0 rounded-full"
                      style={{ backgroundColor: color.background }}
                      aria-hidden="true"
                    />
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2">
                        <span className="truncate font-medium text-navy-900">
                          {person.name || person.track_id.slice(0, 8)}
                        </span>
                        <Badge status={person.status === 'known' ? 'known' : 'anonymous'}>
                          {person.status === 'known' ? 'Bilinen' : 'Anonim'}
                        </Badge>
                      </div>
                      <div className="mt-1 flex flex-wrap items-center gap-x-3 text-xs text-navy-500">
                        <span className="inline-flex items-center gap-1">
                          <Clock className="h-3 w-3" />
                          {formatDurationNs(person.total_duration_ns)}
                        </span>
                        <span>{person.appearance_count} görünüm</span>
                        <span>{person.detection_count} tespit</span>
                      </div>
                      <div className="mt-2 flex items-center gap-2">
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation()
                            onSeekToTrack(person.track_id, person.first_pts_ns)
                          }}
                          className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-primary hover:bg-primary-50 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                        >
                          <Eye className="h-3 w-3" />
                          İlk ana git
                        </button>
                        {person.status !== 'known' && (
                          <Link
                            to={`/enroll/${person.face_id}`}
                            className="inline-flex items-center gap-1 rounded px-2 py-1 text-xs font-medium text-success hover:bg-successSubtle focus:outline-none focus-visible:ring-2 focus-visible:ring-success"
                            onClick={(e) => e.stopPropagation()}
                          >
                            <UserPlus className="h-3 w-3" />
                            Adlandır
                          </Link>
                        )}
                      </div>
                    </div>
                  </div>
                </div>
              </li>
            )
          })}
        </ul>
      </CardContent>
    </Card>
  )
}
