import { VideoDropzone } from '@/components/video/VideoDropzone'
import { VideoPlayer } from '@/components/video/VideoPlayer'
import { Alert } from '@/components/ui/Alert'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { useToast } from '@/hooks/useToast'
import { RotateCcw } from 'lucide-react'
import { useEffect, useState } from 'react'

export default function VideoPage() {
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [playbackError, setPlaybackError] = useState<string | null>(null)
  const { addToast } = useToast()

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      setPlaybackError(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    setPlaybackError(null)
    return () => URL.revokeObjectURL(url)
  }, [file])

  function handleFileChange(next: File | null) {
    setPlaybackError(null)
    setFile(next)
  }

  function handleReset() {
    setFile(null)
    setPreviewUrl(null)
    setPlaybackError(null)
    addToast({ variant: 'info', title: 'Sıfırlandı', message: 'Yeni bir video seçebilirsiniz.' })
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div>
          <h1 className="page-title">Video Tanıma</h1>
          <p className="page-subtitle mt-1">
            Bir video yükleyin; backend tamamlandığında kutu ve kimlik overlay'i burada gösterilecek.
          </p>
        </div>
        {file && (
          <Button type="button" variant="secondary" onClick={handleReset}>
            <RotateCcw className="mr-2 h-4 w-4" aria-hidden="true" />
            Yeni Video
          </Button>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card className="overflow-hidden">
            <CardContent className="space-y-5 p-5">
              <VideoDropzone value={file} onChange={handleFileChange} label="Video Seçin" />
              <p className="text-xs leading-relaxed text-navy-400">
                Yüklenen video şu an yalnızca tarayıcıda önizlenir; sunucuya gönderilmez.
              </p>
            </CardContent>
          </Card>
        </div>

        <div className="lg:col-span-2">
          <Card className="h-full">
            <CardContent className="h-full p-0">
              <VideoPlayer
                src={previewUrl ?? undefined}
                fileName={file?.name}
                className="aspect-video w-full"
                onError={(message) => setPlaybackError(message)}
              />
            </CardContent>
          </Card>
        </div>
      </div>

      {playbackError && (
        <Alert variant="error" title="Video oynatılamadı">
          {playbackError}
        </Alert>
      )}

      {file && !playbackError && (
        <div className="rounded-lg border border-amber-200 bg-amber-50 p-4 text-sm text-amber-800">
          Backend video pipeline tamamlanmadan önce bu sayfa sadece yerel oynatma içindir. İleride yükleme sonrası
          işlem durumu ve bbox/identity overlay burada gösterilecek.
        </div>
      )}
    </div>
  )
}
