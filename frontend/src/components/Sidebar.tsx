import { NavLink } from 'react-router-dom'
import {
  LayoutDashboard,
  Scan,
  Play,
  Users,
  ListVideo,
  BarChart3,
  Settings,
} from 'lucide-react'

const nav = [
  { to: '/', label: 'Overview', icon: LayoutDashboard },
  { to: '/image-test', label: 'Image Test', icon: Scan },
  { to: '/video-test', label: 'Video Test', icon: Play },
  { to: '/faces', label: 'Faces', icon: Users },
  { to: '/processes', label: 'Processes', icon: ListVideo },
  { to: '/analytics', label: 'Analytics', icon: BarChart3 },
]

function Sidebar({ open }: { open: boolean }) {
  return (
    <aside
      className={`flex flex-col border-r border-[var(--color-border)] bg-[var(--color-panel)] transition-all duration-200 ${
        open ? 'w-56' : 'w-16'
      }`}
    >
      <div className="flex h-16 items-center gap-3 border-b border-[var(--color-border)] px-4">
        <img src="/logo.svg" alt="MergenVision" className="h-9 w-9" />
        {open && (
          <div>
            <p className="text-sm font-semibold text-[var(--color-foreground)]">MergenVision</p>
            <p className="text-[10px] uppercase tracking-wider text-[var(--color-dim)]">Operations</p>
          </div>
        )}
      </div>
      <nav className="flex-1 overflow-y-auto p-3 space-y-1">
        {nav.map((item) => {
          const Icon = item.icon
          return (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `group flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-all ${
                  isActive
                    ? 'bg-[var(--color-primary)]/10 text-[var(--color-primary)]'
                    : 'text-[var(--color-muted)] hover:bg-[var(--color-elevated)] hover:text-[var(--color-foreground)]'
                } ${open ? '' : 'justify-center'}`
              }
            >
              <Icon size={20} />
              {open && <span>{item.label}</span>}
            </NavLink>
          )
        })}
      </nav>
      <div className="border-t border-[var(--color-border)] p-3">
        <button
          type="button"
          className={`flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-[var(--color-muted)] transition-colors hover:bg-[var(--color-elevated)] hover:text-[var(--color-foreground)] ${
            open ? '' : 'justify-center'
          }`}
        >
          <Settings size={20} />
          {open && <span>Settings</span>}
        </button>
      </div>
    </aside>
  )
}

export default Sidebar
