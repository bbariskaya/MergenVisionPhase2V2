import { useFace, useFaceHistory } from '@/api/faces'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapRecognizeStatus } from '@/lib/utils'
import { CalendarDays, User } from 'lucide-react'
import { Link, useParams } from 'react-router'

export default function FaceDetailPage() {
  const { faceId } = useParams<{ faceId: string }>()
  const faceQuery = useFace(faceId || '')
  const historyQuery = useFaceHistory(faceId || '')

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

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">{face.name ?? 'İsimsiz yüz'}</h1>
        <p className="page-subtitle mt-1">Kişi kaydı detayı ve tanıma geçmişi.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <Card className="lg:col-span-1">
          <CardContent className="p-5">
            <div className="relative aspect-square overflow-hidden rounded-xl border border-navy-200 bg-navy-50">
              <div className="flex h-full w-full items-center justify-center text-navy-300">
                <User className="h-20 w-20" />
              </div>
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
