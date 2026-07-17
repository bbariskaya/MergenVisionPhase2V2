import {
  useAddFaceSampleMutation,
  useDeleteFaceMutation,
  useDeleteFaceSampleMutation,
  useFace,
  useFaceHistory,
  useFaceSamples,
} from '@/api/faces'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Button } from '@/components/ui/Button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapRecognizeStatus } from '@/lib/utils'
import { CalendarDays, Plus, Trash2, User, X } from 'lucide-react'
import { useRef, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router'

export default function FaceDetailPage() {
  const { faceId } = useParams<{ faceId: string }>()
  const navigate = useNavigate()
  const fileInputRef = useRef<HTMLInputElement>(null)
  const [actionError, setActionError] = useState<string | null>(null)

  const faceQuery = useFace(faceId || '')
  const historyQuery = useFaceHistory(faceId || '')
  const samplesQuery = useFaceSamples(faceId || '')
  const addSample = useAddFaceSampleMutation()
  const deleteSample = useDeleteFaceSampleMutation()
  const deleteFace = useDeleteFaceMutation()

  if (faceQuery.isLoading) {
    return <FaceDetailSkeleton />
  }

  if (faceQuery.error || !faceQuery.data) {
    return (
      <Alert variant="error" title="Yüz kaydı bulunamadı">
        İstenen kayda ulaşılamadı. Kayıt silinmiş olabilir.
      </Alert>
    )
  }

  const face = faceQuery.data
  const firstSampleImage = samplesQuery.data?.samples.at(0)?.image_url

  function handleFileChange(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !faceId) return
    setActionError(null)
    addSample.mutate(
      { faceId, image: file },
      {
        onError: (err) => setActionError(err.message),
        onSuccess: () => {
          if (fileInputRef.current) fileInputRef.current.value = ''
        },
      },
    )
  }

  function handleDeleteSample(sampleId: string) {
    if (!faceId) return
    if (!window.confirm('Bu fotoğrafı silmek istediğinize emin misiniz?')) return
    setActionError(null)
    deleteSample.mutate(
      { faceId, sampleId },
      { onError: (err) => setActionError(err.message) },
    )
  }

  function handleDeletePerson() {
    if (!faceId) return
    if (!window.confirm('Bu kişiyi ve tüm fotoğraflarını silmek istediğinize emin misiniz?')) return
    setActionError(null)
    deleteFace.mutate(faceId, {
      onError: (err) => setActionError(err.message),
      onSuccess: () => navigate('/people'),
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="page-title">{face.name ?? 'İsimsiz yüz'}</h1>
          <p className="page-subtitle mt-1">Kişi kaydı detayı ve tanıma geçmişi.</p>
        </div>
        <Button
          variant="danger"
          size="sm"
          onClick={handleDeletePerson}
          isLoading={deleteFace.isPending}
          className="w-full sm:w-auto"
        >
          <Trash2 className="mr-2 h-4 w-4" />
          Kişiyi Sil
        </Button>
      </div>

      {actionError && (
        <Alert variant="error" title="İşlem başarısız">
          {actionError}
        </Alert>
      )}

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardContent className="p-5">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-navy-200 bg-navy-50">
              {firstSampleImage ? (
                <img
                  src={firstSampleImage}
                  alt={`${face.name ?? 'İsimsiz yüz'} örnek fotoğrafı`}
                  className="h-full w-full object-cover object-center"
                />
              ) : (
                <div className="flex h-full w-full items-center justify-center text-navy-300">
                  <User className="h-20 w-20" />
                </div>
              )}
            </div>
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Durum</span>
                <Badge status={face.status}>{mapRecognizeStatus(face.status)}</Badge>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">face_id</span>
                <span className="truncate pl-2 text-xs font-medium text-navy-900">{face.face_id}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Kayıt Tarihi</span>
                <span className="text-sm font-medium text-navy-900">{formatDate(face.created_at)}</span>
              </div>
              <div className="flex items-center justify-between">
                <span className="text-sm text-navy-500">Güncelleme</span>
                <span className="text-sm font-medium text-navy-900">{formatDate(face.updated_at)}</span>
              </div>
            </div>
          </CardContent>
        </Card>

        <div className="lg:col-span-2">
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Metadata</CardTitle>
            </CardHeader>
            <CardContent>
              {face.metadata ? (
                <pre className="max-h-64 overflow-auto rounded-lg bg-navy-50 p-3 text-xs text-navy-700">
                  {JSON.stringify(face.metadata, null, 2)}
                </pre>
              ) : (
                <p className="text-sm text-navy-500">Metadata yok.</p>
              )}
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
              <CardTitle className="text-base">Yüz Örnekleri</CardTitle>
              <div>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  className="hidden"
                  onChange={handleFileChange}
                  aria-label="Yeni fotoğraf ekle"
                />
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => fileInputRef.current?.click()}
                  isLoading={addSample.isPending}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  Fotoğraf Ekle
                </Button>
              </div>
            </CardHeader>
            <CardContent>
              {samplesQuery.isLoading ? (
                <div className="grid grid-cols-3 gap-3">
                  <Skeleton className="aspect-square w-full rounded-lg" />
                  <Skeleton className="aspect-square w-full rounded-lg" />
                  <Skeleton className="aspect-square w-full rounded-lg" />
                </div>
              ) : samplesQuery.error ? (
                <Alert variant="error" title="Örnekler alınamadı">
                  {samplesQuery.error.message}
                </Alert>
              ) : samplesQuery.data && samplesQuery.data.samples.length > 0 ? (
                <div className="grid grid-cols-3 gap-3 sm:grid-cols-4">
                  {samplesQuery.data.samples.map((sample) => (
                    <div
                      key={sample.sample_id}
                      className="group relative aspect-square overflow-hidden rounded-lg border border-navy-100 bg-navy-50"
                    >
                      {sample.image_url ? (
                        <>
                          <img
                            src={sample.image_url}
                            alt="Kaydedilmiş yüz örneği"
                            className="h-full w-full object-cover object-center"
                            loading="lazy"
                          />
                          <a
                            href={sample.image_url}
                            target="_blank"
                            rel="noreferrer"
                            className="absolute inset-0 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                            aria-label="Örnek görüntüyü aç"
                          />
                        </>
                      ) : (
                        <div className="flex h-full w-full items-center justify-center text-navy-300">
                          <User className="h-8 w-8" aria-hidden="true" />
                        </div>
                      )}
                      <button
                        type="button"
                        onClick={() => handleDeleteSample(sample.sample_id)}
                        disabled={deleteSample.isPending}
                        className="absolute right-1.5 top-1.5 rounded-full bg-white/90 p-1 text-navy-700 opacity-0 shadow-sm transition-opacity hover:bg-danger hover:text-white group-hover:opacity-100 focus-visible:opacity-100 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary disabled:opacity-50"
                        aria-label="Fotoğrafı sil"
                      >
                        <X className="h-3.5 w-3.5" aria-hidden="true" />
                      </button>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-navy-500">Bu kimlik için kaydedilmiş örnek görüntü yok.</p>
              )}
            </CardContent>
          </Card>

          <Card className="mt-6">
            <CardHeader>
              <CardTitle className="text-base">Tanıma Geçmişi</CardTitle>
            </CardHeader>
            <CardContent>
              {historyQuery.isLoading ? (
                <div className="space-y-2">
                  <Skeleton className="h-10 w-full" />
                  <Skeleton className="h-10 w-full" />
                </div>
              ) : historyQuery.error ? (
                <Alert variant="error" title="Geçmiş alınamadı">
                  {historyQuery.error.message}
                </Alert>
              ) : historyQuery.data && historyQuery.data.history.length > 0 ? (
                <ul className="divide-y divide-navy-100">
                  {historyQuery.data.history.map((entry) => (
                    <li key={entry.process_id} className="flex items-center justify-between py-3">
                      <div className="flex items-center gap-3">
                        <CalendarDays className="h-4 w-4 text-navy-400" aria-hidden="true" />
                        <div>
                          <p className="text-sm font-medium text-navy-900">{formatDate(entry.timestamp)}</p>
                          <p className="text-xs text-navy-500">İşlem {entry.process_id.slice(0, 8)}…</p>
                        </div>
                      </div>
                      <Link
                        to={`/processes/${entry.process_id}`}
                        className="text-sm font-medium text-primary hover:underline"
                      >
                        Detay
                      </Link>
                    </li>
                  ))}
                </ul>
              ) : (
                <p className="text-sm text-navy-500">Bu kişi için henüz bir tanıma işlemi yok.</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function FaceDetailSkeleton() {
  return (
    <div className="space-y-4">
      <Skeleton className="h-8 w-48" />
      <Skeleton className="h-64 w-full" />
      <Skeleton className="h-40 w-full" />
    </div>
  )
}
