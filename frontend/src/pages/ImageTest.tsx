import { useState } from 'react'
import { Upload, Scan } from 'lucide-react'
import StatusBadge from '../components/StatusBadge'
import JsonViewer from '../components/JsonViewer'
import { imageResult } from '../mocks/data'

function ImageTest() {
  const [result, setResult] = useState(imageResult)
  const [threshold, setThreshold] = useState(0.75)

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Image Test</h2>
        <p className="text-sm text-[var(--color-muted)]">Upload an image and validate recognition results</p>
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="space-y-5 lg:col-span-2">
          <div className="card flex flex-col items-center justify-center p-8 text-center">
            <Upload className="mx-auto h-10 w-10 text-[var(--color-dim)]" />
            <p className="mt-3 text-sm font-medium text-[var(--color-foreground)]">Drop image here or click to upload</p>
            <p className="text-xs text-[var(--color-dim)]">PNG, JPG up to 10 MB</p>
            <button
              type="button"
              className="btn-primary mt-4 px-4 py-2 text-sm"
            >
              Select Image
            </button>
          </div>

          <div className="card p-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Preview & Overlay</h3>
              <button
                type="button"
                onClick={() => setResult({ ...result })}
                className="btn-primary flex items-center gap-1.5 px-3 py-1.5 text-sm"
              >
                <Scan size={16} />
                Run Recognition
              </button>
            </div>
            <div className="relative mt-4 overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]">
              <img
                src="https://picsum.photos/seed/mergen-image/1280/720"
                alt="Preview"
                className="w-full opacity-90"
              />
              {result.faces.map((face) => {
                const bb = face.boundingBox
                const color =
                  face.status === 'known'
                    ? 'var(--color-primary)'
                    : face.status === 'anonymous'
                    ? '#94a3b8'
                    : 'var(--color-accent)'
                return (
                  <div
                    key={face.faceId}
                    className="absolute border-2"
                    style={{
                      left: `${(bb.x / 1280) * 100}%`,
                      top: `${(bb.y / 720) * 100}%`,
                      width: `${(bb.width / 1280) * 100}%`,
                      height: `${(bb.height / 720) * 100}%`,
                      borderColor: color,
                    }}
                  >
                    <span
                      className="absolute -top-6 left-0 whitespace-nowrap rounded px-1.5 py-0.5 text-[10px] font-bold text-white"
                      style={{ backgroundColor: color }}
                    >
                      {face.name || `${face.faceId} • ${(face.confidence * 100).toFixed(0)}%`}
                    </span>
                  </div>
                )
              })}
            </div>
          </div>
        </div>

        <div className="space-y-5">
          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Configuration</h3>
            <div className="mt-4">
              <label htmlFor="threshold" className="text-xs font-medium uppercase tracking-wider text-[var(--color-dim)]">
                Confidence threshold
              </label>
              <input
                id="threshold"
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="mt-3 w-full accent-[var(--color-primary)]"
              />
              <p className="text-right text-xs text-[var(--color-muted)]">{threshold.toFixed(2)}</p>
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">
              Detected Faces ({result.faceCount})
            </h3>
            <div className="mt-4 space-y-3">
              {result.faces.map((face) => (
                <div key={face.faceId} className="rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-3">
                  <div className="flex items-center justify-between">
                    <p className="text-sm font-medium text-[var(--color-foreground)]">{face.faceId}</p>
                    <StatusBadge status={face.status} />
                  </div>
                  {face.name && <p className="text-xs text-[var(--color-muted)]">{face.name}</p>}
                  <p className="text-xs text-[var(--color-dim)]">Confidence: {(face.confidence * 100).toFixed(1)}%</p>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Raw JSON</h3>
            <div className="mt-4">
              <JsonViewer data={result} />
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default ImageTest
