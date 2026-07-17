import { useHealth } from '@/api/health'
import { cn } from '@/lib/utils'
import { Activity, AlertCircle, CheckCircle2, Loader2 } from 'lucide-react'

export function HealthIndicator() {
  const health = useHealth()
  const isLoading = health.isLoading
  const isOk = health.data?.status === 'ok'

  const statusClasses = isLoading
    ? 'bg-navy-50 text-navy-500 border-navy-200'
    : isOk
      ? 'bg-successSubtle text-success border-emerald-200'
      : 'bg-dangerSubtle text-danger border-red-200'

  return (
    <button
      className={cn(
        'inline-flex items-center gap-2 rounded-full border px-3 py-1.5 text-xs font-medium transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary focus-visible:ring-offset-2',
        statusClasses,
      )}
      aria-label="Sistem durumu"
    >
      {isLoading ? (
        <Loader2 className="h-3.5 w-3.5 animate-spin" aria-hidden="true" />
      ) : isOk ? (
        <CheckCircle2 className="h-3.5 w-3.5" aria-hidden="true" />
      ) : (
        <AlertCircle className="h-3.5 w-3.5" aria-hidden="true" />
      )}
      <span>{isLoading ? 'Kontrol' : isOk ? 'Hazır' : 'Hata'}</span>
      <Activity className="h-3.5 w-3.5 opacity-60" aria-hidden="true" />
    </button>
  )
}
