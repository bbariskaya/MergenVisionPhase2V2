import { useMemo, useState } from 'react'
import { Search, Trash2, Edit3, UserPlus, Filter } from 'lucide-react'
import StatusBadge from '../components/StatusBadge'
import { faces } from '../mocks/data'

function Faces() {
  const [query, setQuery] = useState('')
  const [statusFilter, setStatusFilter] = useState('all')

  const filtered = useMemo(() => {
    return faces.filter((face) => {
      const matchesQuery =
        (face.name || '').toLowerCase().includes(query.toLowerCase()) ||
        face.id.toLowerCase().includes(query.toLowerCase())
      const matchesStatus = statusFilter === 'all' || face.status === statusFilter
      return matchesQuery && matchesStatus
    })
  }, [query, statusFilter])

  return (
    <div className="space-y-5">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Faces</h2>
          <p className="text-sm text-[var(--color-muted)]">Manage known and anonymous face identities</p>
        </div>
        <button
          type="button"
          className="btn-primary flex items-center gap-2 px-4 py-2 text-sm"
        >
          <UserPlus size={16} />
          Enroll Face
        </button>
      </div>

      <div className="card p-4">
        <div className="flex flex-col gap-3 sm:flex-row">
          <div className="relative flex-1">
            <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-dim)]" size={16} />
            <input
              type="text"
              placeholder="Search by name or ID..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              className="h-10 w-full rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] pl-9 pr-3 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)] placeholder:text-[var(--color-dim)]"
            />
          </div>
          <div className="relative">
            <Filter className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-dim)]" size={16} />
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="h-10 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] pl-9 pr-3 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)]"
            >
              <option value="all">All statuses</option>
              <option value="known">Known</option>
              <option value="anonymous">Anonymous</option>
              <option value="new_anonymous">New Anonymous</option>
            </select>
          </div>
        </div>
      </div>

      <div className="overflow-hidden rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)] shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-sm">
            <thead className="bg-[var(--color-bg)] text-[var(--color-dim)]">
              <tr>
                <th className="px-4 py-3 font-medium">ID</th>
                <th className="px-4 py-3 font-medium">Name</th>
                <th className="px-4 py-3 font-medium">Status</th>
                <th className="px-4 py-3 font-medium">Metadata</th>
                <th className="px-4 py-3 font-medium">Samples</th>
                <th className="px-4 py-3 font-medium">Last Seen</th>
                <th className="px-4 py-3 font-medium text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-[var(--color-border)]">
              {filtered.map((face) => (
                <tr key={face.id} className="transition-colors hover:bg-[var(--color-elevated)]/50">
                  <td className="px-4 py-3 font-mono text-[var(--color-foreground)]">{face.id}</td>
                  <td className="px-4 py-3 text-[var(--color-foreground)]">{face.name || <span className="text-[var(--color-dim)]">—</span>}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={face.status} />
                  </td>
                  <td className="px-4 py-3 text-xs text-[var(--color-muted)]">
                    {Object.entries(face.metadata).map(([k, v]) => `${k}: ${v}`).join(', ') || '—'}
                  </td>
                  <td className="px-4 py-3 text-[var(--color-foreground)]">{face.sampleCount}</td>
                  <td className="px-4 py-3 text-[var(--color-muted)]">{new Date(face.lastSeen).toLocaleString()}</td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-2">
                      <button type="button" className="rounded p-1.5 text-[var(--color-muted)] hover:bg-[var(--color-elevated)]">
                        <Edit3 size={16} />
                      </button>
                      <button type="button" className="rounded p-1.5 text-[var(--color-alert)] hover:bg-[var(--color-alert)]/10">
                        <Trash2 size={16} />
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {filtered.length === 0 && (
          <div className="p-8 text-center text-sm text-[var(--color-muted)]">No faces found.</div>
        )}
      </div>
    </div>
  )
}

export default Faces
