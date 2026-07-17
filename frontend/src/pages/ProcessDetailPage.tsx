import { useProcess } from '@/api/processes'
import { Alert } from '@/components/ui/Alert'
import { Badge } from '@/components/ui/Badge'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/Card'
import { Skeleton } from '@/components/ui/Skeleton'
import { formatDate, mapProcessStatus, mapRecognizeStatus } from '@/lib/utils'
import { Link, useParams } from 'react-router'

export default function ProcessDetailPage() {
  const { processId } = useParams<{ processId: string }>()
  const processQuery = useProcess(processId || '')

  if (processQuery.isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <Skeleton className="h-40 w-full" />
      </div>
    )
  }

  if (processQuery.error || !processQuery.data) {
    return (
      <Alert variant="error" title="İşlem bulunamadı">
        İstenen tanıma işlemine ulaşılamadı.
      </Alert>
    )
  }

  const proc = processQuery.data
  const detections = proc.details?.detections

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">İşlem Detayı</h1>
        <p className="page-subtitle mt-1">{proc.process_id}</p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Genel Bilgiler</CardTitle>
        </CardHeader>
        <CardContent>
          <dl className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <div>
              <dt className="text-sm text-navy-500">Durum</dt>
              <dd>
                <Badge status={proc.status}>{mapProcessStatus(proc.status)}</Badge>
              </dd>
            </div>
            <div>
              <dt className="text-sm text-navy-500">İşlem Tipi</dt>
              <dd className="font-medium text-navy-900">{proc.process_type}</dd>
            </div>
            <div>
              <dt className="text-sm text-navy-500">Yüz Sayısı</dt>
              <dd className="font-medium text-navy-900">{proc.face_count ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-sm text-navy-500">Hata Kodu</dt>
              <dd className="font-medium text-navy-900">{proc.error_code ?? '—'}</dd>
            </div>
            <div>
              <dt className="text-sm text-navy-500">Başlangıç</dt>
              <dd className="font-medium text-navy-900">{formatDate(proc.created_at)}</dd>
            </div>
            <div>
              <dt className="text-sm text-navy-500">Bitiş</dt>
              <dd className="font-medium text-navy-900">{proc.completed_at ? formatDate(proc.completed_at) : '—'}</dd>
            </div>
          </dl>
        </CardContent>
      </Card>

      {detections && detections.length > 0 && (
        <>
          <h2 className="text-lg font-semibold text-navy-900">Tespit Edilen Yüzler</h2>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
            {detections.map((face) => (
              <Card key={face.face_id}>
                <CardContent className="p-5">
                  <div className="mb-3 flex items-center justify-between">
                    <span className="font-semibold text-navy-900">{face.name ?? 'İsimsiz yüz'}</span>
                    <Badge status={face.status}>{mapRecognizeStatus(face.status)}</Badge>
                  </div>
                  {face.confidence !== null && (
                    <p className="text-sm text-navy-500">Güven: {Math.round(face.confidence * 100)}%</p>
                  )}
                  {face.face_id ? (
                    <Link
                      to={`/faces/${face.face_id}`}
                      className="mt-3 inline-block text-sm font-medium text-primary hover:underline"
                    >
                      Yüz Detayını Gör
                    </Link>
                  ) : (
                    <p className="mt-3 text-sm text-slate-400">Kayıtlı yüz eşleşmesi yok.</p>
                  )}
                </CardContent>
              </Card>
            ))}
          </div>
        </>
      )}

      {detections && detections.length === 0 && (
        <Alert variant="info" title="Yüz bulunamadı">
          Bu işlemde herhangi bir yüz tespit edilmedi.
        </Alert>
      )}
    </div>
  )
}
