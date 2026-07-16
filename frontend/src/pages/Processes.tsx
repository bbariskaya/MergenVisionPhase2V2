import { useMemo, useState } from 'react'
import { Search, Image as ImageIcon, Film } from 'lucide-react'
import StatusBadge from '../components/StatusBadge'
import JsonViewer from '../components/JsonViewer'
import { processes, videoJob } from '../mocks/data'

function Processes() {
  const [query, setQuery] = useState('')
  const [typeFilter, setTypeFilter] = useState<'all' | 'image' | 'video'>('all')

  const filtered = useMemo(() => {
    return processes.filter((p) => {
      const matchesQuery = p.processId.toLowerCase().includes(query.toLowerCase())
      const matchesType = typeFilter === 'all' || p.type === typeFilter
      return matchesQuery && matchesType
    })
  }, [query, typeFilter])

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Processes & Jobs</h2>
        <p className="text-sm text-[var(--color-muted)]">Trace image and video recognition jobs</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-2">
          <div className="card p-4">
            <div className="flex flex-col gap-3 sm:flex-row">
              <div className="relative flex-1">
                <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-dim)]" size={16} />
                <input
                  type="text"
                  placeholder="Search process or job ID..."
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  className="h-10 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] pl-9 pr-3 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)] placeholder:text-[var(--color-dim)]"
                />
              </div>
              <select
                value={typeFilter}
                onChange={(e) => setTypeFilter(e.target.value as 'all' | 'image' | 'video')}
                className="h-10 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] px-3 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)]"
              >
                <option value="all">All types</option>
                <option value="image">Image</option>
                <option value="video">Video</option>
              </select>
            </div>
          </div>

          <div className="overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left text-sm">
                <thead className="bg-[var(--color-bg)] text-[var(--color-dim)]">
                  <tr>
                    <th className="px-4 py-3 font-medium">Process / Job ID</th>
                    <th className="px-4 py-3 font-medium">Type</th>
                    <th className="px-4 py-3 font-medium">Detected</th>
                    <th className="px-4 py-3 font-medium">Duration</th>
                    <th className="px-4 py-3 font-medium">Status</th>
                    <th className="px-4 py-3 font-medium">Timestamp</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[var(--color-border)]">
                  {filtered.map((p) => (
                    <tr key={p.processId} className="transition-colors hover:bg-[var(--color-elevated)]/50">
                      <td className="px-4 py-3 font-mono font-medium text-[var(--color-primary)]">{p.processId}</td>
                      <td className="px-4 py-3 text-[var(--color-foreground)]">
                        <div className="flex items-center gap-2">
                          {p.type === 'image' ? <ImageIcon size={16} /> : <Film size={16} />}
                          <span className="capitalize">{p.type}</span>
                        </div>
                      </td>
                      <td className="px-4 py-3 text-[var(--color-foreground)]">
                        {p.type === 'video' ? `${p.personCount ?? 0} persons` : `${p.faceCount} faces`}
                      </td>
                      <td className="px-4 py-3 text-[var(--color-muted)]">{(p.durationMs / 1000).toFixed(2)}s</td>
                      <td className="px-4 py-3">
                        <StatusBadge status={p.status} />
                      </td>
                      <td className="px-4 py-3 text-[var(--color-muted)]">{new Date(p.timestamp).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Selected Job Detail</h3>
            <div className="mt-4 space-y-3 text-sm">
              {[
                ['Job ID', videoJob.jobId],
                ['Process ID', videoJob.processId],
                ['Duration', `${videoJob.duration}s`],
                ['Resolution', `${videoJob.width}×${videoJob.height}`],
                ['Processed frames', `${videoJob.processedFrames}`],
                ['Persons', `${videoJob.personCount}`],
              ].map(([k, v]) => (
                <div key={k} className="flex justify-between">
                  <span className="text-[var(--color-muted)]">{k}</span>
                  <span className={`font-mono text-[var(--color-foreground)] ${k === 'Status' ? '' : ''}`}>{v}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Raw Process JSON</h3>
            <div className="mt-4">
              <JsonViewer data={videoJob} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Processes
