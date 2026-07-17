import { useRecognizeMutation } from '@/api/faces'
import type { RecognitionFace, RecognizeResponse } from '@/api/types'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { FileDropzone } from '@/components/ui/FileDropzone'
import { useToast } from '@/hooks/useToast'
import { cn, formatConfidence, isAnonymousStatus, mapRecognizeStatus, statusColor } from '@/lib/utils'
import { ImageIcon, Loader2, RotateCcw, ScanFace, Search, UserCheck } from 'lucide-react'
import { useEffect, useRef, useState } from 'react'
import { Link } from 'react-router'

export default function IdentifyPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<RecognizeResponse | null>(null)

  const { addToast } = useToast()
  const recognize = useRecognizeMutation()

  const imageRef = useRef<HTMLImageElement>(null)
  const [imageSize, setImageSize] = useState<{ width: number; height: number; naturalWidth: number; naturalHeight: number } | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      setResult(null)
      setImageSize(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!file) return
    setResult(null)
    try {
      const data = await recognize.mutateAsync({ image: file })
      setResult(data)
      if (data.face_count === 0) {
        addToast({ variant: 'info', title: 'Yüz algılanmadı', message: 'Görselde tanımlanabilir yüz bulunamadı.' })
      } else {
        addToast({ variant: 'success', title: 'Tanıma tamamlandı', message: `${data.face_count} yüz bulundu.` })
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Tanıma işlemi başarısız.'
      addToast({ variant: 'error', title: 'Tanıma hatası', message })
    }
  }

  function reset() {
    setFile(null)
    setPreviewUrl(null)
    setResult(null)
    setImageSize(null)
    recognize.reset()
  }

  function updateImageSize() {
    const img = imageRef.current
    if (!img) return
    setImageSize({
      width: img.clientWidth,
      height: img.clientHeight,
      naturalWidth: img.naturalWidth,
      naturalHeight: img.naturalHeight,
    })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="page-title">Yüz Tanıma</h1>
          <p className="page-subtitle mt-1">Bir görsel yükleyerek kayıtlı kişilerle eşleştirin.</p>
        </div>
        {result && (
          <Button type="button" variant="secondary" onClick={reset}>
            <RotateCcw className="mr-2 h-4 w-4" />
            Yeni Görsel
          </Button>
        )}
      </div>

      <form onSubmit={handleSubmit} className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card className="overflow-hidden">
            <CardContent className="space-y-5 p-5">
              <FileDropzone value={file} onChange={setFile} previewUrl={previewUrl} label="Görsel Yükle" />
              <Button type="submit" className="w-full" isLoading={recognize.isPending} disabled={!file}>
                <ScanFace className="mr-2 h-4 w-4" />
                {recognize.isPending ? 'Tanıma yapılıyor…' : 'Yüzleri Tanı'}
              </Button>
              <p className="text-xs leading-relaxed text-navy-400">
                Yüklenen görseller sadece tanıma işlemi için kullanılır ve saklanmaz.
              </p>
            </CardContent>
          </Card>

          {result && <ResultSummary result={result} />}
        </div>

        <div className="lg:col-span-2">
          <Card className="h-full min-h-[360px]">
            <CardContent className="relative h-full p-0">
              {recognize.isPending ? (
                <div className="relative flex h-full min-h-[360px] items-center justify-center overflow-hidden rounded-xl">
                  {previewUrl && (
                    <img
                      src={previewUrl}
                      alt="Sorgu görseli"
                      className="absolute inset-0 h-full w-full object-contain opacity-40"
                    />
                  )}
                  <div className="relative z-10 flex flex-col items-center text-navy-600">
                    <Loader2 className="mb-3 h-10 w-10 animate-spin text-primary" aria-hidden="true" />
                    <p className="font-medium">Yüzler tanınıyor…</p>
                    <p className="text-sm">Bu işlem birkaç saniye sürebilir.</p>
                  </div>
                </div>
              ) : previewUrl ? (
                <div className="relative mx-auto inline-block max-w-full p-4">
                  <img
                    ref={imageRef}
                    src={previewUrl}
                    alt="Sorgu görseli"
                    className="block max-h-[70vh] w-auto rounded-lg"
                    onLoad={updateImageSize}
                  />
                  {result && imageSize && result.faces.map((face, i) => (
                    <FaceOverlay
                      key={face.face_id}
                      face={face}
                      index={i}
                      imageSize={imageSize}
                    />
                  ))}
                </div>
              ) : (
                <div className="flex min-h-[360px] flex-col items-center justify-center text-navy-400">
                  <div className="mb-3 rounded-full bg-navy-50 p-3 text-navy-300">
                    <ImageIcon className="h-8 w-8" aria-hidden="true" />
                  </div>
                  <p className="font-medium text-navy-600">Başlamak için sol taraftan bir görsel yükleyin.</p>
                  <p className="text-sm">JPEG veya PNG, maksimum 10 MB.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </form>

      {result && result.faces.length === 0 && (
        <Alert variant="info" title="Görselde yüz bulunamadı">
          Lütfen net, ön cephe bir yüz görseli yükleyin.
        </Alert>
      )}

      {result && result.faces.length > 0 && (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {result.faces.map((face, i) => (
            <FaceResultCard key={face.face_id} face={face} index={i} />
          ))}
        </div>
      )}
    </div>
  )
}

function ResultSummary({ result }: { result: RecognizeResponse }) {
  const known = result.faces.filter((f) => f.status === 'known').length
  const anonymous = result.faces.filter((f) => isAnonymousStatus(f.status)).length
  return (
    <Card>
      <CardContent className="space-y-4 p-5">
        <h3 className="font-semibold text-navy-900">Sonuç Özeti</h3>
        <div className="grid grid-cols-3 gap-2 text-center">
          <div className="rounded-lg bg-navy-50 p-2">
            <p className="text-lg font-bold text-navy-900">{result.face_count}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Yüz</p>
          </div>
          <div className="rounded-lg bg-successSubtle p-2">
            <p className="text-lg font-bold text-success">{known}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Bilinen</p>
          </div>
          <div className="rounded-lg bg-navy-50 p-2">
            <p className="text-lg font-bold text-navy-900">{anonymous}</p>
            <p className="text-[10px] uppercase tracking-wide text-navy-500">Bilinmeyen</p>
          </div>
        </div>
        <Link to={`/processes/${result.process_id}`} className="btn-secondary block w-full px-3 py-2 text-center text-xs">
          İşlem Detayını Gör
        </Link>
      </CardContent>
    </Card>
  )
}

function FaceOverlay({
  face,
  index,
  imageSize,
}: {
  face: RecognitionFace
  index: number
  imageSize: { width: number; height: number; naturalWidth: number; naturalHeight: number }
}) {
  const { x, y, width, height } = face.bounding_box
  const scaleX = imageSize.width / imageSize.naturalWidth
  const scaleY = imageSize.height / imageSize.naturalHeight
  const left = x * scaleX
  const top = y * scaleY
  const w = width * scaleX
  const h = height * scaleY

  return (
    <div
      className="absolute border-2 border-primary bg-primary/10"
      style={{ left, top, width: w, height: h }}
      aria-label={`Yüz ${index + 1}`}
    >
      <span className="absolute -top-6 left-0 rounded bg-primary px-1.5 py-0.5 text-xs font-bold text-white">
        {index + 1}
      </span>
    </div>
  )
}

function FaceResultCard({ face, index }: { face: RecognitionFace; index: number }) {
  const status = face.status
  const isKnown = status === 'known'

  return (
    <div className="card p-5">
      <div className="mb-4 flex items-center justify-between">
        <span className="font-semibold text-navy-900">Yüz {index + 1}</span>
        <StatusBadge status={status} />
      </div>

      <div className="flex items-center gap-3">
        <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-navy-50 text-navy-400">
          {isKnown ? <UserCheck className="h-6 w-6" /> : <Search className="h-6 w-6" />}
        </div>
        <div className="min-w-0 flex-1">
          <p className="truncate font-medium text-navy-900">{face.name ?? 'Bilinmeyen kişi'}</p>
          {face.confidence !== null && (
            <p className="text-xs text-navy-500">Güven: {formatConfidence(face.confidence)}</p>
          )}
        </div>
      </div>

      {isKnown ? (
        <Link
          to={`/faces/${face.face_id}`}
          className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
        >
          Kişi detayını gör
        </Link>
      ) : (
        <Link
          to={`/enroll/${face.face_id}`}
          className="mt-4 inline-flex items-center gap-1 text-sm font-medium text-primary hover:underline"
        >
          Kaydet (enroll)
        </Link>
      )}
    </div>
  )
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={cn(
        'inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-xs font-medium',
        statusColor(status),
      )}
    >
      {status === 'known' ? <UserCheck className="h-3 w-3" /> : <Search className="h-3 w-3" />}
      {mapRecognizeStatus(status)}
    </span>
  )
}
