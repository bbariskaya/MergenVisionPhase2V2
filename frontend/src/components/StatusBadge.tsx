function StatusBadge({ status }: { status: 'known' | 'anonymous' | 'new_anonymous' | string }) {
  const styles: Record<string, string> = {
    known: 'bg-[var(--color-primary)]/10 text-[var(--color-primary)] border border-[var(--color-primary)]/20',
    anonymous: 'bg-[var(--color-dim)]/10 text-[var(--color-muted)] border border-[var(--color-border-light)]',
    new_anonymous: 'bg-[var(--color-accent)]/10 text-[var(--color-accent)] border border-[var(--color-accent)]/20',
    completed: 'bg-[var(--color-success)]/10 text-[var(--color-success)] border border-[var(--color-success)]/20',
    processing: 'bg-[var(--color-secondary)]/10 text-[var(--color-secondary)] border border-[var(--color-secondary)]/20',
    pending: 'bg-[var(--color-warning)]/10 text-[var(--color-warning)] border border-[var(--color-warning)]/20',
    failed: 'bg-[var(--color-alert)]/10 text-[var(--color-alert)] border border-[var(--color-alert)]/20',
  }

  return (
    <span
      className={`inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-semibold ${
        styles[status] || styles.anonymous
      }`}
    >
      {status}
    </span>
  )
}

export default StatusBadge
