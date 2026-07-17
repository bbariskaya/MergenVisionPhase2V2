import { cn } from '@/lib/utils'
import { Clapperboard, Upload, X } from 'lucide-react'
import { type ChangeEvent, type DragEvent, useRef, useState } from 'react'
import { Button } from '../ui/Button'

export interface VideoDropzoneProps {
  accept?: string
  maxSizeBytes?: number
  value: File | null
  onChange: (file: File | null) => void
  label?: string
  helperText?: string
  error?: string
}

export function VideoDropzone({
  accept = 'video/*',
  maxSizeBytes,
  value,
  onChange,
  label = 'Video Yükle',
  helperText = maxSizeBytes
    ? 'Sürükleyip bırakın veya tıklayın. Desteklenen video dosyaları.'
    : 'Sürükleyip bırakın veya tıklayın. Yerel oynatma için boyut sınırı yok.',
  error,
}: VideoDropzoneProps) {
  const inputRef = useRef<HTMLInputElement>(null)
  const [isDragging, setIsDragging] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  function validateAndSet(file: File) {
    if (!file.type.startsWith('video/')) {
      setLocalError('Yalnızca video dosyaları kabul edilir.')
      return
    }
    if (maxSizeBytes !== undefined && Number.isFinite(maxSizeBytes) && file.size > maxSizeBytes) {
      setLocalError('Dosya boyutu limiti aşıyor.')
      return
    }
    setLocalError(null)
    onChange(file)
  }

  function handleFileSelect(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (file) validateAndSet(file)
  }

  function handleDrop(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files?.[0]
    if (file) validateAndSet(file)
  }

  function handleDragOver(e: DragEvent<HTMLDivElement>) {
    e.preventDefault()
    setIsDragging(true)
  }

  function handleDragLeave() {
    setIsDragging(false)
  }

  function clear() {
    onChange(null)
    setLocalError(null)
    if (inputRef.current) inputRef.current.value = ''
  }

  const displayedError = error || localError

  return (
    <div className="w-full">
      {label && <p className="label">{label}</p>}
      {value ? (
        <div className="relative overflow-hidden rounded-xl border border-navy-200 bg-white">
          <div className="flex items-center gap-3 p-4">
            <div className="rounded-lg bg-primary-50 p-2 text-primary">
              <Clapperboard className="h-6 w-6" aria-hidden="true" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-navy-900">{value.name}</p>
              <p className="text-xs text-navy-500">{(value.size / (1024 * 1024)).toFixed(2)} MB</p>
            </div>
            <Button type="button" variant="danger" size="sm" onClick={clear} aria-label="Videoyu kaldır">
              <X className="h-4 w-4" aria-hidden="true" />
            </Button>
          </div>
        </div>
      ) : (
        <div
          onClick={() => inputRef.current?.click()}
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={cn(
            'flex cursor-pointer flex-col items-center justify-center rounded-xl border-2 border-dashed bg-navy-50 p-8 transition-colors duration-150 hover:bg-navy-100',
            isDragging ? 'border-primary bg-primary-50' : 'border-navy-300',
            displayedError && 'border-danger bg-red-50',
          )}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
          aria-label={label}
        >
          <Upload className={cn('mb-2 h-8 w-8', displayedError ? 'text-danger' : 'text-navy-400')} aria-hidden="true" />
          <p className="text-sm font-medium text-navy-700">
            {isDragging ? 'Buraya bırakın' : 'Video dosyası seçin veya sürükleyin'}
          </p>
          <p className="mt-1 text-xs text-navy-500">{helperText}</p>
          <input
            ref={inputRef}
            type="file"
            accept={accept}
            className="sr-only"
            onChange={handleFileSelect}
            aria-hidden="true"
          />
        </div>
      )}
      {displayedError && (
        <p className="mt-1.5 text-xs text-danger" role="alert">
          {displayedError}
        </p>
      )}
    </div>
  )
}
