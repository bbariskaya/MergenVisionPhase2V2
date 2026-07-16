import { useEffect } from 'react'
import { Outlet } from 'react-router-dom'
import Sidebar from './Sidebar'
import Header from './Header'

function Layout() {
  useEffect(() => {
    document.documentElement.classList.add('dark')
    document.body.style.backgroundColor = 'var(--color-bg)'
  }, [])

  return (
    <div className="flex h-screen w-full overflow-hidden bg-[var(--color-bg)]">
      <Sidebar open={true} />
      <div className="flex flex-1 flex-col min-w-0">
        <Header />
        <main className="flex-1 overflow-auto p-4 lg:p-5">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default Layout
