import { Link } from 'react-router-dom'
import {
  Scan,
  Play,
  Users,
  BarChart3,
  ListVideo,
  Activity,
  ShieldCheck,
  ChevronRight,
  Upload,
  Plus,
  Cpu,
  Activity as ActivityIcon,
} from 'lucide-react'
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
} from 'recharts'
import StatCard from '../components/StatCard'
import { faces, analyticsData } from '../mocks/data'

const modules = [
  { label: 'Face Recognition', to: '/faces', icon: Users, color: '#10b981' },
  { label: 'Video Analytics', to: '/video-test', icon: Play, color: '#0ea5e9' },
  { label: 'Image Analysis', to: '/image-test', icon: Scan, color: '#a855f7' },
  { label: 'Analytics', to: '/analytics', icon: BarChart3, color: '#f97316' },
]

const cameras = [
  { id: 'CAM 01', label: 'Lobby Entrance', seed: 'cam01', boxes: [{ x: 38, y: 28, w: 18, h: 34 }] },
  { id: 'CAM 02', label: 'Parking Entry', seed: 'cam02', boxes: [{ x: 25, y: 35, w: 45, h: 28 }] },
  { id: 'CAM 03', label: 'Warehouse Aisle 3', seed: 'cam03', boxes: [{ x: 55, y: 22, w: 12, h: 42 }] },
  { id: 'CAM 04', label: 'Reception', seed: 'cam04', boxes: [{ x: 42, y: 38, w: 16, h: 22 }] },
  { id: 'CAM 05', label: 'Outdoor Perimeter', seed: 'cam05', boxes: [{ x: 33, y: 25, w: 14, h: 40 }] },
  { id: 'CAM 06', label: 'Loading Dock', seed: 'cam06', boxes: [{ x: 18, y: 20, w: 60, h: 45 }] },
]

const events = [
  { id: 1, type: 'Known person', label: 'Lobby Entrance – CAM 01', confidence: 0.94, time: '10:24:18', color: 'var(--color-success)', icon: Users },
  { id: 2, type: 'Unknown face', label: 'Reception – CAM 04', confidence: 0.42, time: '10:23:57', color: 'var(--color-alert)', icon: Users },
  { id: 3, type: 'Motion detected', label: 'Parking Entry – CAM 02', confidence: 0.78, time: '10:23:41', color: 'var(--color-accent)', icon: Activity },
  { id: 4, type: 'Known person', label: 'Warehouse Aisle 3 – CAM 03', confidence: 0.91, time: '10:23:10', color: 'var(--color-success)', icon: Users },
  { id: 5, type: 'Unknown face', label: 'Outdoor Perimeter – CAM 05', confidence: 0.38, time: '10:22:48', color: 'var(--color-alert)', icon: Users },
  { id: 6, type: 'Job completed', label: 'Video job job_8f3c1a2e', confidence: null, time: '10:22:26', color: 'var(--color-secondary)', icon: Play },
]

function Dashboard() {
  const knownCount = faces.filter((f) => f.status === 'known').length
  const anonCount = faces.length - knownCount

  return (
    <div className="space-y-5">
      <div className="flex items-center gap-3">
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Global overview</h2>
        <div className="flex items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-elevated)] px-3 py-1 text-xs font-medium text-[var(--color-muted)]">
          <span className="h-2 w-2 rounded-full bg-[var(--color-success)]" />
          All systems operational
        </div>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Live streams" value="12" sub="2 alerts" icon={ListVideo} accent="secondary" />
        <StatCard title="Video jobs" value="7" sub="67% processed" icon={Play} accent="primary">
          <div className="mt-3 h-1.5 w-full overflow-hidden rounded-full bg-[var(--color-elevated)]">
            <div className="h-full rounded-full bg-[var(--color-primary)]" style={{ width: '67%' }} />
          </div>
        </StatCard>
        <StatCard title="Events today" value="1,284" sub="+12% vs yesterday" icon={Activity} accent="accent" />
        <StatCard title="Service health" value="98.7%" sub="All nodes healthy" icon={ShieldCheck} accent="success" />
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        {modules.map((m) => {
          const Icon = m.icon
          return (
            <Link
              key={m.label}
              to={m.to}
              className="card group flex items-center justify-between p-4 transition-colors hover:bg-[var(--color-elevated)]"
            >
              <div className="flex items-center gap-3">
                <div className="rounded-lg p-2" style={{ backgroundColor: `${m.color}15`, color: m.color }}>
                  <Icon size={22} />
                </div>
                <span className="text-sm font-semibold text-[var(--color-foreground)]">{m.label}</span>
              </div>
              <ChevronRight size={18} className="text-[var(--color-dim)] transition-transform group-hover:translate-x-0.5" />
            </Link>
          )
        })}
      </div>

      <div className="grid gap-5 lg:grid-cols-3">
        <div className="lg:col-span-2 space-y-4">
          <div className="card p-4">
            <div className="mb-4 flex items-center justify-between">
              <div className="flex items-center gap-3">
                <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Live streams</h3>
                <div className="flex items-center gap-1.5 text-xs text-[var(--color-muted)]">
                  <span className="h-2 w-2 rounded-full bg-[var(--color-alert)]" />
                  12 live
                </div>
              </div>
              <div className="flex gap-2">
                <button className="rounded p-1 text-[var(--color-dim)] hover:bg-[var(--color-elevated)]">
                  <BarChart3 size={16} />
                </button>
              </div>
            </div>
            <div className="grid gap-3 sm:grid-cols-2 xl:grid-cols-3">
              {cameras.map((cam) => (
                <div key={cam.id} className="relative overflow-hidden rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)]">
                  <img
                    src={`https://picsum.photos/seed/${cam.seed}/400/225`}
                    alt={cam.label}
                    className="h-36 w-full object-cover opacity-80"
                  />
                  <div className="absolute inset-0">
                    {cam.boxes.map((box, idx) => (
                      <div
                        key={idx}
                        className="absolute border-2 border-[var(--color-success)]"
                        style={{
                          left: `${box.x}%`,
                          top: `${box.y}%`,
                          width: `${box.w}%`,
                          height: `${box.h}%`,
                        }}
                      />
                    ))}
                  </div>
                  <div className="absolute left-2 top-2 flex items-center gap-1.5 rounded bg-black/60 px-2 py-0.5 text-[10px] font-medium text-white">
                    <span className="h-1.5 w-1.5 rounded-full bg-[var(--color-success)]" />
                    {cam.id} – {cam.label}
                  </div>
                  <div className="absolute bottom-2 left-2 rounded bg-[var(--color-alert)] px-1.5 py-0.5 text-[10px] font-bold text-white">
                    LIVE
                  </div>
                  <div className="absolute bottom-2 right-2 rounded bg-black/60 px-1.5 py-0.5 text-[10px] text-white">
                    10:24:32
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Process Volume</h3>
            <div className="mt-4 h-60">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analyticsData.processVolume}>
                  <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                  <XAxis dataKey="date" stroke="var(--color-muted)" fontSize={12} />
                  <YAxis stroke="var(--color-muted)" fontSize={12} />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: 'var(--color-surface)',
                      borderColor: 'var(--color-border)',
                      color: 'var(--color-foreground)',
                    }}
                  />
                  <Bar dataKey="image" fill="var(--color-primary)" radius={[4, 4, 0, 0]} />
                  <Bar dataKey="video" fill="var(--color-secondary)" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        <div className="space-y-4">
          <div className="card p-4">
            <div className="mb-3 flex items-center justify-between">
              <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Recent events</h3>
              <Link to="/processes" className="text-xs font-medium text-[var(--color-primary)] hover:underline">
                View all
              </Link>
            </div>
            <div className="space-y-3">
              {events.map((ev) => {
                const Icon = ev.icon
                return (
                  <div
                    key={ev.id}
                    className="flex items-center gap-3 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] p-2.5 transition-colors hover:border-[var(--color-border-light)]"
                  >
                    <img
                      src={`https://picsum.photos/seed/ev${ev.id}/48/48`}
                      alt=""
                      className="h-10 w-10 rounded-md object-cover"
                    />
                    <div className="min-w-0 flex-1">
                      <p className="text-xs font-semibold" style={{ color: ev.color }}>
                        {ev.type}
                      </p>
                      <p className="truncate text-xs text-[var(--color-muted)]">{ev.label}</p>
                      {ev.confidence !== null && (
                        <p className="text-[10px] text-[var(--color-dim)]">Confidence: {(ev.confidence * 100).toFixed(0)}%</p>
                      )}
                    </div>
                    <div className="text-right">
                      <p className="text-[10px] font-mono text-[var(--color-dim)]">{ev.time}</p>
                      <Icon size={16} style={{ color: ev.color }} className="ml-auto mt-1" />
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card p-4">
            <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Faces Registry</h3>
            <div className="mt-4 space-y-3">
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-muted)]">Known</span>
                <span className="font-semibold text-[var(--color-success)]">{knownCount}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-elevated)]">
                <div
                  className="h-full rounded-full bg-[var(--color-success)]"
                  style={{ width: `${(knownCount / faces.length) * 100}%` }}
                />
              </div>
              <div className="flex items-center justify-between text-sm">
                <span className="text-[var(--color-muted)]">Anonymous</span>
                <span className="font-semibold text-[var(--color-alert)]">{anonCount}</span>
              </div>
              <div className="h-2 w-full overflow-hidden rounded-full bg-[var(--color-elevated)]">
                <div
                  className="h-full rounded-full bg-[var(--color-alert)]"
                  style={{ width: `${(anonCount / faces.length) * 100}%` }}
                />
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="card flex flex-wrap items-center gap-4 p-3">
        <Link
          to="/image-test"
          className="btn-primary flex items-center gap-2 px-4 py-2.5 text-sm"
        >
          <Upload size={16} />
          Analyze media
        </Link>
        <Link
          to="/video-test"
          className="flex items-center gap-2 rounded-lg border border-[var(--color-border)] bg-[var(--color-elevated)] px-4 py-2.5 text-sm font-medium text-[var(--color-foreground)] hover:bg-[var(--color-border-light)]"
        >
          <Plus size={16} />
          Start video job
        </Link>
        <div className="ml-auto flex flex-wrap items-center gap-6 text-xs text-[var(--color-muted)]">
          <div className="flex items-center gap-2">
            <Cpu size={16} className="text-[var(--color-primary)]" />
            <span>GPU Usage</span>
            <div className="h-1.5 w-20 overflow-hidden rounded-full bg-[var(--color-elevated)]">
              <div className="h-full rounded-full bg-[var(--color-primary)]" style={{ width: '62%' }} />
            </div>
            <span className="font-mono text-[var(--color-foreground)]">62%</span>
          </div>
          <div className="flex items-center gap-2">
            <ActivityIcon size={16} className="text-[var(--color-secondary)]" />
            <span>API Latency</span>
            <span className="font-mono text-[var(--color-foreground)]">128 ms</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="h-2 w-2 rounded-full bg-[var(--color-success)]" />
            <span>API Healthy</span>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Dashboard
