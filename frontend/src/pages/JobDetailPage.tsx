import {
  buildPlaybackUrl,
  useCancelVideoJobMutation,
  useRetryVideoMutation,
  useVideo,
  useVideoAppearances,
  useVideoJob,
  useVideoJobResult,
  useVideoOverlayFrames,
  useVideoPeople,
} from '@/api/videos'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { EmptyState } from '@/components/ui/EmptyState'
import { Skeleton } from '@/components/ui/Skeleton'
import { AppearanceTimeline } from '@/components/video/AppearanceTimeline'
import { TrackListPanel } from '@/components/video/TrackListPanel'
import { VideoOverlayPlayer } from '@/components/video/VideoOverlayPlayer'
import { useToast } from '@/hooks/useToast'
import { cn } from '@/lib/utils'
import { formatDurationNs, ptsToSeconds } from '@/lib/video'
import { Ban, Eye, EyeOff, Film, RefreshCw, Tag, User, Users } from 'lucide-react'
import { useEffect, useMemo, useRef, useState } from 'react'
import { useNavigate, useParams } from 'react-router'

export default function JobDetailPage() {
  const { videoId, jobId } = useParams<{ videoId: string; jobId: string }>()
  const videoRef = useRef<HTMLVideoElement>(null)
  const navigate = useNavigate()
  const { addToast } = useToast()

  const [selectedTrackId, setSelectedTrackId] = useState<string | null>(null)
  const [filterKnown, setFilterKnown] = useState(true)
  const [filterAnonymous, setFilterAnonymous] = useState(true)
  const [showOverlay, setShowOverlay] = useState(true)
  const [showLabels, setShowLabels] = useState(true)
  const [seekTime, setSeekTime] = useState<number | null>(null)
  const [playbackTime, setPlaybackTime] = useState(0)

  const videoQuery = useVideo(videoId ?? '')
  const jobQuery = useVideoJob(jobId ?? '')
  const resultQuery = useVideoJobResult(jobId ?? '')
  const peopleQuery = useVideoPeople(jobId ?? '')
  const appearancesQuery = useVideoAppearances(jobId ?? '')
  const framesQuery = useVideoOverlayFrames(
    jobId ?? '',
    0,
    undefined,
    resultQuery.data?.result_available === true,
  )

  const cancelMutation = useCancelVideoJobMutation()
  const retryMutation = useRetryVideoMutation()

  useEffect(() => {
    if (seekTime !== null && videoRef.current) {
      videoRef.current.currentTime = seekTime
      setSeekTime(null)
    }
  }, [seekTime])

  const job = jobQuery.data
  const isActive = job && ['pending', 'processing', 'cancelling'].includes(job.state)
  const isCompleted = job?.state === 'completed'
  const isFailed = job?.state === 'failed'
  const isCancelled = job?.state === 'cancelled'
  const durationSeconds = useMemo(() => {
    const ns = videoQuery.data?.duration_ns
    return ns !== null && ns !== undefined ? ptsToSeconds(ns) : 0
  }, [videoQuery.data])

  function handleCancel() {
    if (!jobId) return
    if (!window.confirm('İşlemi iptal etmek istediğinize emin misiniz?')) return
    cancelMutation.mutate(jobId, {
      onSuccess: () => addToast({ variant: 'info', title: 'İptal istendi', message: 'İşlem iptali backend’e iletildi.' }),
      onError: () => addToast({ variant: 'error', title: 'İptal başarısız', message: 'İptal isteği gönderilemedi.' }),
    })
  }

  function handleRetry() {
    if (!jobId) return
    retryMutation.mutate(
      { jobId, idempotencyKey: crypto.randomUUID() },
      {
        onSuccess: (data) => {
          navigate(`/videos/${data.video_id}/jobs/${data.job_id}`)
          addToast({ variant: 'success', title: 'Yeniden başlatıldı', message: 'Yeni job oluşturuldu.' })
        },
        onError: () => addToast({ variant: 'error', title: 'Yeniden başlatma başarısız', message: 'Job tekrar kuyruğa alınamadı.' }),
      },
    )
  }

  if (!videoId || !jobId) {
    return (
      <Alert variant="error" title="Geçersiz adres">
        Video veya job bilgisi eksik.
      </Alert>
    )
  }

  if (jobQuery.isLoading || videoQuery.isLoading) {
    return <JobSkeleton />
  }

  if (jobQuery.isError) {
    return (
      <Alert variant="error" title="Job yüklenemedi">
        {jobQuery.error?.message || 'İşlem bilgisi alınamadı.'}
      </Alert>
    )
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <h1 className="page-title truncate">{job ? jobFileName(job) : 'Video İşlem'}</h1>
            {job && <JobStateBadge state={job.state} />}
          </div>
          <p className="page-subtitle mt-1">
            {videoQuery.data && (
              <>
                {formatMediaResolution(videoQuery.data.display_width, videoQuery.data.display_height)} ·{' '}
                {formatDurationNs(videoQuery.data.duration_ns ?? 0)} · {videoQuery.data.container_format || '—'}
              </>
            )}
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          {isActive && (
            <Button variant="danger" onClick={handleCancel} isLoading={cancelMutation.isPending}>
              <Ban className="mr-2 h-4 w-4" />
              İptal Et
            </Button>
          )}
          {(isFailed || isCancelled) && (
            <Button onClick={handleRetry} isLoading={retryMutation.isPending}>
              <RefreshCw className="mr-2 h-4 w-4" />
              Tekrar Dene
            </Button>
          )}
          <Button variant="secondary" onClick={() => navigate('/videos')}>
            Videolar
          </Button>
        </div>
      </div>

      {job && <JobProgressCard job={job} />}

      {isFailed && job.error_code && (
        <Alert variant="error" title="İşlem başarısız oldu">
          Kod: <code className="rounded bg-red-100 px-1 py-0.5 text-xs">{job.error_code}</code>
          <div className="mt-1">Job ID: <code className="text-xs">{job.job_id}</code></div>
        </Alert>
      )}

      {isCancelled && (
        <Alert variant="warning" title="İşlem iptal edildi">
          Kullanıcı tarafından iptal edildi.
        </Alert>
      )}

      {isActive && (
        <Alert variant="info" title="İşlem devam ediyor">
          Video GPU pipeline tarafından işleniyor. Tamamlandığında overlay ve kimlik listesi otomatik görünecek.
        </Alert>
      )}

      <div className={cn('grid gap-4 lg:grid-cols-3', isCompleted ? 'lg:grid-rows-1' : '')}>
        <div className="space-y-4 lg:col-span-2">
          {isCompleted && videoQuery.data ? (
            <>
              <OverlayControls
                filterKnown={filterKnown}
                filterAnonymous={filterAnonymous}
                showOverlay={showOverlay}
                showLabels={showLabels}
                onToggleKnown={() => setFilterKnown((v) => !v)}
                onToggleAnonymous={() => setFilterAnonymous((v) => !v)}
                onToggleOverlay={() => setShowOverlay((v) => !v)}
                onToggleLabels={() => setShowLabels((v) => !v)}
              />
              <VideoOverlayPlayer
                ref={videoRef}
                src={buildPlaybackUrl(videoId)}
                fileName={jobFileName(job)}
                displayWidth={videoQuery.data.display_width ?? 640}
                displayHeight={videoQuery.data.display_height ?? 480}
                rotationDegrees={videoQuery.data.rotation_degrees}
                durationSeconds={durationSeconds}
                frames={framesQuery.data?.frames ?? []}
                people={peopleQuery.data?.people ?? []}
                appearances={appearancesQuery.data?.appearances ?? []}
                selectedTrackId={selectedTrackId}
                showOverlay={showOverlay}
                filterKnown={filterKnown}
                filterAnonymous={filterAnonymous}
                showLabels={showLabels}
                onPlaybackTimeChange={setPlaybackTime}
                onError={(message) => addToast({ variant: 'error', title: 'Oynatma hatası', message })}
              />
            </>
          ) : (
            <Card className="flex min-h-[360px] items-center justify-center">
              <EmptyState
                icon={Film}
                title="Video hazır değil"
                description="İşlem tamamlandığında orijinal video ve overlay burada gösterilecek."
              />
            </Card>
          )}
        </div>

        <div className="lg:col-span-1">
          <TrackListPanel
            people={peopleQuery.data?.people ?? []}
            selectedTrackId={selectedTrackId}
            filterKnown={filterKnown}
            filterAnonymous={filterAnonymous}
            isLoading={peopleQuery.isLoading}
            onSelectTrack={setSelectedTrackId}
            onSeekToTrack={(_, ptsNs) => setSeekTime(ptsToSeconds(ptsNs))}
          />
        </div>
      </div>

      {isCompleted && (
        <AppearanceTimeline
          people={peopleQuery.data?.people ?? []}
          appearances={appearancesQuery.data?.appearances ?? []}
          durationSeconds={durationSeconds}
          currentTimeSeconds={playbackTime}
          selectedTrackId={selectedTrackId}
          onSelectTrack={setSelectedTrackId}
          onSeek={(seconds) => setSeekTime(seconds)}
        />
      )}
    </div>
  )
}

function JobSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-10 w-1/2" />
      <Skeleton className="h-8 w-full" />
      <div className="grid gap-4 lg:grid-cols-3">
        <Skeleton className="aspect-video lg:col-span-2" />
        <Skeleton className="h-96 lg:col-span-1" />
      </div>
    </div>
  )
}

function JobProgressCard({ job }: { job: { state: string; stage: string; progress_percent: number; processed_frames: number; sampled_frames: number; detected_observations: number; person_count: number } }) {
  return (
    <Card>
      <CardContent className="p-4">
        <div className="mb-2 flex items-center justify-between text-sm">
          <span className="font-medium text-navy-700">{job.stage}</span>
          <span className="tabular-nums text-navy-500">%{job.progress_percent}</span>
        </div>
        <div className="h-2 w-full overflow-hidden rounded-full bg-navy-100">
          <div
            className="h-full rounded-full bg-primary transition-all duration-300"
            style={{ width: `${job.progress_percent}%` }}
            aria-hidden="true"
          />
        </div>
        <div className="mt-3 grid grid-cols-2 gap-2 text-xs text-navy-500 sm:grid-cols-4">
          <span>İşlenen frame: {job.processed_frames}</span>
          <span>Örneklenen frame: {job.sampled_frames}</span>
          <span>Tespit: {job.detected_observations}</span>
          <span>Kişi: {job.person_count}</span>
        </div>
      </CardContent>
    </Card>
  )
}

function JobStateBadge({ state }: { state: string }) {
  const styles: Record<string, string> = {
    pending: 'bg-amber-100 text-amber-800 border-amber-200',
    processing: 'bg-blue-100 text-blue-800 border-blue-200',
    cancelling: 'bg-amber-100 text-amber-800 border-amber-200',
    completed: 'bg-emerald-100 text-emerald-800 border-emerald-200',
    failed: 'bg-red-100 text-red-800 border-red-200',
    cancelled: 'bg-slate-100 text-slate-700 border-slate-200',
  }
  const labels: Record<string, string> = {
    pending: 'Bekliyor',
    processing: 'İşleniyor',
    cancelling: 'İptal Ediliyor',
    completed: 'Tamamlandı',
    failed: 'Hata',
    cancelled: 'İptal Edildi',
  }
  return (
    <span className={cn('rounded-full border px-2.5 py-0.5 text-xs font-medium', styles[state] || styles.pending)}>
      {labels[state] || state}
    </span>
  )
}

function OverlayControls({
  filterKnown,
  filterAnonymous,
  showOverlay,
  showLabels,
  onToggleKnown,
  onToggleAnonymous,
  onToggleOverlay,
  onToggleLabels,
}: {
  filterKnown: boolean
  filterAnonymous: boolean
  showOverlay: boolean
  showLabels: boolean
  onToggleKnown: () => void
  onToggleAnonymous: () => void
  onToggleOverlay: () => void
  onToggleLabels: () => void
}) {
  return (
    <Card>
      <CardContent className="flex flex-wrap items-center gap-2 p-3">
        <ControlButton active={showOverlay} onClick={onToggleOverlay} icon={<Eye className="h-4 w-4" />} label="Overlay" />
        <ControlButton active={showLabels} onClick={onToggleLabels} icon={<Tag className="h-4 w-4" />} label="Etiketler" />
        <ControlButton active={filterKnown} onClick={onToggleKnown} icon={<User className="h-4 w-4" />} label="Bilinen" />
        <ControlButton
          active={filterAnonymous}
          onClick={onToggleAnonymous}
          icon={<Users className="h-4 w-4" />}
          label="Anonim"
        />
      </CardContent>
    </Card>
  )
}

function ControlButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean
  onClick: () => void
  icon: React.ReactNode
  label: string
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={cn(
        'inline-flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-xs font-medium transition-colors duration-150 focus:outline-none focus-visible:ring-2 focus-visible:ring-primary',
        active
          ? 'border-primary bg-primary-50 text-primary'
          : 'border-navy-200 bg-white text-navy-600 hover:bg-navy-50',
      )}
      aria-pressed={active}
    >
      {active ? icon : <EyeOff className="h-4 w-4" />}
      {label}
    </button>
  )
}

function jobFileName(job: { job_id: string } | null): string {
  return job ? `Job ${job.job_id.slice(0, 8)}` : 'Video'
}

function formatMediaResolution(width: number | null | undefined, height: number | null | undefined): string {
  if (width && height) return `${width}×${height}`
  return '—'
}
