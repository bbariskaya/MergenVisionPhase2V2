import {
  Activity,
  BarChart3,
  PieChart as PieChartIcon,
  TrendingUp,
} from 'lucide-react'
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  CartesianGrid,
  PieChart,
  Pie,
  Cell,
  BarChart,
  Bar,
} from 'recharts'
import StatCard from '../components/StatCard'
import { analyticsData, faces, processes } from '../mocks/data'

const statusColors: Record<string, string> = {
  Known: '#10b981',
  Anonymous: '#64748b',
  'New Anonymous': '#f97316',
}

const jobHealthData = [
  { name: 'Completed', value: analyticsData.jobHealth.completed },
  { name: 'Failed', value: analyticsData.jobHealth.failed },
  { name: 'Processing', value: analyticsData.jobHealth.processing },
  { name: 'Pending', value: analyticsData.jobHealth.pending },
]

const jobColors: Record<string, string> = {
  Completed: '#22c55e',
  Failed: '#ef4444',
  Processing: '#0ea5e9',
  Pending: '#eab308',
}

function Analytics() {
  const knownPct = Math.round((faces.filter((f) => f.status === 'known').length / faces.length) * 100)

  return (
    <div className="space-y-5">
      <div>
        <h2 className="text-xl font-semibold text-[var(--color-foreground)]">Analytics</h2>
        <p className="text-sm text-[var(--color-muted)]">Recognition trends and system health</p>
      </div>

      <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
        <StatCard title="Recognition Accuracy" value="94%" sub="Based on confidence threshold" icon={TrendingUp} accent="success" />
        <StatCard title="Known Faces" value={`${knownPct}%`} sub={`${faces.length} total registered`} icon={PieChartIcon} accent="primary" />
        <StatCard title="Total Jobs" value={processes.length} sub="Last 7 days" icon={BarChart3} accent="secondary" />
        <StatCard title="Avg Confidence" value="0.87" sub="+0.03 vs last week" icon={Activity} accent="accent" />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <div className="card p-4">
          <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Confidence Trend</h3>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={analyticsData.confidenceTrend}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis dataKey="time" stroke="var(--color-muted)" fontSize={12} />
                <YAxis domain={[0.6, 1]} stroke="var(--color-muted)" fontSize={12} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface)',
                    borderColor: 'var(--color-border)',
                    color: 'var(--color-foreground)',
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="confidence"
                  stroke="var(--color-primary)"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="card p-4">
          <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Face Status Distribution</h3>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <PieChart>
                <Pie
                  data={analyticsData.statusDistribution}
                  dataKey="value"
                  nameKey="name"
                  innerRadius={60}
                  outerRadius={80}
                  paddingAngle={4}
                >
                  {analyticsData.statusDistribution.map((entry) => (
                    <Cell key={entry.name} fill={statusColors[entry.name]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface)',
                    borderColor: 'var(--color-border)',
                    color: 'var(--color-foreground)',
                  }}
                />
              </PieChart>
            </ResponsiveContainer>
          </div>
          <div className="mt-2 flex flex-wrap justify-center gap-4 text-xs">
            {analyticsData.statusDistribution.map((entry) => (
              <div key={entry.name} className="flex items-center gap-1.5">
                <span className="h-3 w-3 rounded-full" style={{ backgroundColor: statusColors[entry.name] }} />
                <span className="text-[var(--color-muted)]">{entry.name} ({entry.value})</span>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
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

        <div className="card p-4">
          <h3 className="text-sm font-semibold text-[var(--color-foreground)]">Job Health</h3>
          <div className="mt-4 h-60">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={jobHealthData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border)" />
                <XAxis type="number" stroke="var(--color-muted)" fontSize={12} />
                <YAxis dataKey="name" type="category" stroke="var(--color-muted)" fontSize={12} width={90} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'var(--color-surface)',
                    borderColor: 'var(--color-border)',
                    color: 'var(--color-foreground)',
                  }}
                />
                <Bar dataKey="value" radius={[0, 4, 4, 0]}>
                  {jobHealthData.map((entry) => (
                    <Cell key={entry.name} fill={jobColors[entry.name]} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  )
}

export default Analytics
