import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { cn } from '@/lib/utils'
import { buildTimelineLanes, formatMediaTimeSeconds, formatDurationNs, stableTrackColor } from '@/lib/video'
import { Film } from 'lucide-react'
import { useMemo, useRef } from 'react'
import type { VideoAppearanceEntry, VideoPersonSummary } from '@/api/types'

export interface AppearanceTimelineProps {
  people: VideoPersonSummary[]
  appearances: VideoAppearanceEntry[]
  durationSeconds: number
  currentTimeSeconds: number
  selectedTrackId?: string | null
  onSelectTrack: (trackId: string) => void
  onSeek: (seconds: number) => void
}

export function AppearanceTimeline({
  people,
  appearances,
  durationSeconds,
  currentTimeSeconds,
  selectedTrackId,
  onSelectTrack,
  onSeek,
}: AppearanceTimelineProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const lanes = useMemo(() => buildTimelineLanes(people, appearances), [people, appearances])

  if (durationSeconds <= 0 || lanes.length === 0) {
    return (
      <Card>
        <CardContent className="p-4">
          <EmptyState icon={Film} title="Timeline uygun değil" description="Henüz görünüm aralığı yok." />
        </CardContent>
      </Card>
    )
  }

  const playheadLeft = Math.min(100, Math.max(0, (currentTimeSeconds / durationSeconds) * 100))

  return (
    <Card>
      <CardContent className="space-y-3 p-4">
        <div className="flex items-center justify-between">
          <h2 className="text-sm font-semibold text-navy-900">Görünüm zaman çizelgesi</h2>
          <span className="text-xs tabular-nums text-navy-500">{formatMediaTimeSeconds(currentTimeSeconds)}</span>
        </div>
        <div
          ref={containerRef}
          className="relative overflow-x-auto rounded-lg border border-navy-200 bg-navy-50"
          role="region"
          aria-label="Görünüm zaman çizelgesi"
        >
          <div className="relative min-w-[600px] px-3 py-2">
            <TimeRuler durationSeconds={durationSeconds} />
            <div
              className="pointer-events-none absolute top-0 z-10 h-full w-px bg-danger"
              style={{ left: `${playheadLeft}%` }}
              aria-hidden="true"
            />
            {lanes.map((lane) => {
              const selected = selectedTrackId === lane.trackId
              const color = stableTrackColor(lane.trackId)
              return (
                <div
                  key={lane.trackId}
                  className={cn(
                    'group flex items-center gap-2 border-b border-navy-200 py-1.5 last:border-b-0',
                    selected && 'bg-primary-50',
                  )}
                >
                  <button
                    type="button"
                    onClick={() => onSelectTrack(lane.trackId)}
                    className="shrink-0 truncate text-left text-xs font-medium text-navy-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    style={{ width: '8rem' }}
                    title={lane.name || lane.trackId}
                  >
                    {lane.name || lane.trackId.slice(0, 8)}
                  </button>
                  <div className="relative flex-1">
                    {lane.intervals.map((interval, idx) => {
                      const start = interval.startPtsNs / 1_000_000_000
                      const end = interval.endPtsNs / 1_000_000_000
                      const left = (start / durationSeconds) * 100
                      const width = Math.max(0.5, ((end - start) / durationSeconds) * 100)
                      return (
                        <button
                          key={`${lane.trackId}-${idx}`}
                          type="button"
                          onClick={() => onSeek(start)}
                          className={cn(
                            'absolute h-5 rounded transition-opacity duration-150 hover:opacity-90 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary',
                            selected ? 'opacity-100' : 'opacity-80 group-hover:opacity-100',
                          )}
                          style={{
                            left: `${left}%`,
                            width: `${width}%`,
                            backgroundColor: color.background,
                          }}
                          title={`${lane.name || lane.trackId.slice(0, 8)} · ${formatMediaTimeSeconds(start)} - ${formatMediaTimeSeconds(end)}`}
                          aria-label={`${lane.name || lane.trackId}, ${formatDurationNs(interval.startPtsNs)} ile ${formatDurationNs(interval.endPtsNs)} arasında göründü`}
                        />
                      )
                    })}
                  </div>
                </div>
              )
            })}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function TimeRuler({ durationSeconds }: { durationSeconds: number }) {
  const steps = 10
  const ticks = Array.from({ length: steps + 1 }, (_, i) => (i / steps) * durationSeconds)
  return (
    <div className="relative mb-1 h-4 text-xs text-navy-400">
      {ticks.map((t) => (
        <span
          key={t}
          className="absolute top-0 -translate-x-1/2 tabular-nums"
          style={{ left: `${(t / durationSeconds) * 100}%` }}
        >
          {formatMediaTimeSeconds(t)}
        </span>
      ))}
    </div>
  )
}
