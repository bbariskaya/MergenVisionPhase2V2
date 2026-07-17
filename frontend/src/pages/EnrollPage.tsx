import { useEnrollMutation } from '@/api/faces'
import type { EnrollResponse } from '@/api/types'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { Input } from '@/components/ui/Input'
import { cn } from '@/lib/utils'
import { CheckCircle2, RotateCcw, UserCheck } from 'lucide-react'
import { useState } from 'react'
import { Link, useParams } from 'react-router'

export default function EnrollPage() {
  const { faceId } = useParams<{ faceId: string }>()
  const [name, setName] = useState('')
  const [metadataText, setMetadataText] = useState('')
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<EnrollResponse | null>(null)

  const enroll = useEnrollMutation()

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!faceId || !name.trim()) return

    setError(null)
    let metadata: Record<string, unknown> | undefined
    if (metadataText.trim()) {
      try {
        metadata = JSON.parse(metadataText.trim()) as Record<string, unknown>
      } catch {
        setError('Metadata geçerli bir JSON olmalıdır.')
        return
      }
    }

    try {
      const data = await enroll.mutateAsync({ face_id: faceId, name: name.trim(), metadata })
      setResult(data)
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Kayıt işlemi başarısız.'
      setError(message)
    }
  }

  function reset() {
    setName('')
    setMetadataText('')
    setError(null)
    setResult(null)
    enroll.reset()
  }

  if (result) {
    return (
      <div className="mx-auto max-w-xl space-y-6">
        <Card>
          <CardContent className="p-8 text-center">
            <div className="mb-4 inline-flex h-16 w-16 items-center justify-center rounded-full bg-successSubtle text-success">
              <UserCheck className="h-8 w-8" aria-hidden="true" />
            </div>
            <h1 className="page-title">Kayıt Tamamlandı</h1>
            <p className="page-subtitle mt-1">Yeni yüz kaydı başarıyla oluşturuldu.</p>
          </CardContent>
        </Card>
        <Card>
          <CardContent className="space-y-4 p-6">
            <div className="flex items-center justify-between border-b border-navy-100 py-2">
              <span className="text-sm text-navy-500">Kişi</span>
              <span className="text-sm font-medium text-navy-900">{result.name}</span>
            </div>
            <div className="flex items-center justify-between border-b border-navy-100 py-2">
              <span className="text-sm text-navy-500">face_id</span>
              <span className="text-sm font-medium text-navy-900">{result.face_id}</span>
            </div>
            <div className="flex items-center justify-between border-b border-navy-100 py-2">
              <span className="text-sm text-navy-500">Durum</span>
              <span className="text-sm font-medium text-navy-900">{result.status}</span>
            </div>
            <div className="flex flex-col gap-2 pt-2 sm:flex-row">
              <Button variant="secondary" onClick={reset} className="w-full sm:w-auto">
                <RotateCcw className="mr-2 h-4 w-4" />
                Yeni Kayıt
              </Button>
              <Link to={`/faces/${result.face_id}`} className="btn-primary w-full justify-center sm:w-auto">
                Detayı Gör
              </Link>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <div>
        <h1 className="page-title">Yüz Kaydı</h1>
        <p className="page-subtitle mt-1">Anonim yüzü kayıtlı bir kimliğe yükseltin.</p>
      </div>

      <Card>
        <CardContent className="p-6">
          <form onSubmit={handleSubmit} className="space-y-5">
            <Input
              label="Ad Soyad"
              value={name}
              onChange={(e) => setName(e.target.value)}
              placeholder="Örn: Ahmet Yılmaz"
              required
              autoComplete="name"
            />
            <div>
              <label htmlFor="metadata" className="label">Metadata (isteğe bağlı JSON)</label>
              <textarea
                id="metadata"
                value={metadataText}
                onChange={(e) => setMetadataText(e.target.value)}
                placeholder='{"department": "IT"}'
                className={cn(
                  'input min-h-[120px] font-mono text-sm',
                  error && !enroll.error && 'border-danger focus:border-danger focus:ring-danger/20',
                )}
              />
            </div>
            {error && (
              <Alert variant="error" title="Kayıt başarısız">
                {error}
              </Alert>
            )}
            <Button type="submit" isLoading={enroll.isPending} disabled={!name.trim() || !faceId}>
              <CheckCircle2 className="mr-2 h-4 w-4" />
              Kaydet
            </Button>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
