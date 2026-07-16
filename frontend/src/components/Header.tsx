import { Search, Bell, HelpCircle, User } from 'lucide-react'

function Header() {
  return (
    <header className="flex h-16 items-center justify-between border-b border-[var(--color-border)] bg-[var(--color-panel)] px-4 lg:px-6">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-3">
          <img src="/logo.svg" alt="MergenVision" className="h-7 w-7" />
          <h1 className="hidden text-base font-semibold tracking-wide text-[var(--color-foreground)] sm:block">
            MERGENVISION VISION OPERATIONS
          </h1>
        </div>
        <div className="hidden items-center gap-2 rounded-full border border-[var(--color-border)] bg-[var(--color-elevated)] px-3 py-1 text-xs font-medium text-[var(--color-muted)] md:flex">
          <span className="h-2 w-2 rounded-full bg-[var(--color-success)]" />
          Operational
        </div>
      </div>

      <div className="flex items-center gap-2 lg:gap-4">
        <div className="relative hidden sm:block">
          <Search className="absolute left-2.5 top-1/2 -translate-y-1/2 text-[var(--color-dim)]" size={16} />
          <input
            type="text"
            placeholder="Search faces, jobs, events..."
            className="h-9 w-64 rounded-lg border border-[var(--color-border)] bg-[var(--color-bg)] pl-9 pr-3 text-sm text-[var(--color-foreground)] outline-none focus:ring-1 ring-[var(--color-primary-dim)] placeholder:text-[var(--color-dim)]"
          />
        </div>
        <button type="button" className="relative rounded-lg p-2 text-[var(--color-muted)] hover:bg-[var(--color-elevated)]">
          <Bell size={20} />
          <span className="absolute right-1.5 top-1.5 h-2 w-2 rounded-full bg-[var(--color-alert)]" />
        </button>
        <button type="button" className="rounded-lg p-2 text-[var(--color-muted)] hover:bg-[var(--color-elevated)]">
          <HelpCircle size={20} />
        </button>
        <div className="flex h-9 w-9 items-center justify-center rounded-full border border-[var(--color-border)] bg-[var(--color-elevated)] text-[var(--color-foreground)]">
          <User size={18} />
        </div>
      </div>
    </header>
  )
}

export default Header
