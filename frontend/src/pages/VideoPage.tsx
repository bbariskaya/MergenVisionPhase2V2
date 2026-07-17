import { useUploadVideoMutation } from '@/api/videos'
import { Button } from '@/components/ui/Button'
import { Card, CardContent } from '@/components/ui/Card'
import { VideoDropzone } from '@/components/video/VideoDropzone'
import { useToast } from '@/hooks/useToast'
import { apiErrorMessage } from '@/lib/errors'
import { Film, Loader2, Upload, Video } from 'lucide-react'
import { useEffect, useState } from 'react'
import { useNavigate } from 'react-router'

export default function VideoPage() {
  const navigate = useNavigate()
  const { addToast } = useToast()
  const uploadMutation = useUploadVideoMutation()
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  useEffect(() => {
    if (!file) {
      setPreviewUrl(null)
      return
    }
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
    return () => URL.revokeObjectURL(url)
  }, [file])

  async function handleUpload() {
    if (!file) return
    const idempotencyKey = crypto.randomUUID()
    uploadMutation.mutate(
      { file, idempotencyKey },
      {
        onSuccess: (data) => {
          addToast({ variant: 'success', title: 'Yükleme tamamlandı', message: 'Job kuyruğa alındı.' })
          navigate(`/videos/${data.video_id}/jobs/${data.job_id}`)
        },
        onError: (error) => {
          addToast({ variant: 'error', title: 'Yükleme başarısız', message: apiErrorMessage(error) })
        },
      },
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="page-title">Video Tanıma</h1>
        <p className="page-subtitle mt-1">Bir video yükleyin ve GPU işlem pipeline’ının sonucunu takip edin.</p>
      </div>

      <div className="grid gap-6 lg:grid-cols-3">
        <div className="space-y-4 lg:col-span-1">
          <Card className="overflow-hidden">
            <CardContent className="space-y-5 p-5">
              <VideoDropzone
                value={file}
                onChange={setFile}
                label="Video Seçin"
                helperText="Sürükleyip bırakın veya tıklayın. MP4/MOV/WEBM, container backend tarafından tekrar doğrulanır."
              />

              <Button data-testid="upload-video-button" onClick={handleUpload} disabled={!file || uploadMutation.isPending} isLoading={uploadMutation.isPending} className="w-full">
                {uploadMutation.isPending ? (
                  <>
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                    Yükleniyor…
                  </>
                ) : (
                  <>
                    <Upload className="mr-2 h-4 w-4" />
                    Yükle ve İşle
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <InfoCard />
        </div>

        <div className="lg:col-span-2">
          <Card className="h-full overflow-hidden">
            <CardContent className="h-full p-0">
              {previewUrl ? (
                <video
                  src={previewUrl}
                  controls
                  preload="metadata"
                  className="block aspect-video w-full object-contain"
                  aria-label={file ? `Önizleme: ${file.name}` : 'Video önizleme'}
                />
              ) : (
                <div className="flex aspect-video flex-col items-center justify-center bg-navy-50 text-navy-400">
                  <div className="mb-3 rounded-full bg-navy-100 p-4 text-navy-300">
                    <Video className="h-10 w-10" />
                  </div>
                  <p className="font-medium text-navy-600">Yerel önizleme</p>
                  <p className="text-sm">Soldan bir video seçin.</p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}

function InfoCard() {
  return (
    <Card>
      <CardContent className="space-y-3 p-5 text-sm text-navy-600">
        <div className="flex items-start gap-2">
          <Film className="mt-0.5 h-4 w-4 shrink-0 text-navy-400" />
          <p>Backend videoyu decode eder, doğrular ve kuyruğa alır.</p>
        </div>
        <div className="flex items-start gap-2">
          <Loader2 className="mt-0.5 h-4 w-4 shrink-0 text-navy-400" />
          <p>GPU worker tamamlandığında sonuç sayfasına yönlendirileceksiniz.</p>
        </div>
      </CardContent>
    </Card>
  )
}
