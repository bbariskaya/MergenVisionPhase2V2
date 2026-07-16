import type { LucideIcon } from 'lucide-react'

function StatCard({
  title,
  value,
  sub,
  icon: Icon,
  accent = 'primary',
  children,
}: {
  title: string
  value: string | number
  sub?: string
  icon: LucideIcon
  accent?: 'primary' | 'secondary' | 'accent' | 'alert' | 'success'
  children?: React.ReactNode
}) {
  const accentVar =
    accent === 'secondary'
      ? 'var(--color-secondary)'
      : accent === 'accent'
      ? 'var(--color-accent)'
      : accent === 'alert'
      ? 'var(--color-alert)'
      : accent === 'success'
      ? 'var(--color-success)'
      : 'var(--color-primary)'

  return (
    <div className="card p-4">
      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-[var(--color-dim)]">{title}</p>
          <p className="mt-1.5 text-2xl font-semibold text-[var(--color-foreground)]">{value}</p>
          {sub && <p className="mt-1 text-xs text-[var(--color-muted)]">{sub}</p>}
        </div>
        <div
          className="rounded-lg p-2"
          style={{ backgroundColor: `${accentVar}15`, color: accentVar }}
        >
          <Icon size={20} />
        </div>
      </div>
      {children}
    </div>
  )
}

export default StatCard
