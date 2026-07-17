import { ErrorBoundary } from '@/components/ErrorBoundary'
import { Layout } from '@/components/Layout'
import { ToastContainer } from '@/components/Toast'
import { useToast } from '@/hooks/useToast'
import DashboardPage from '@/pages/DashboardPage'
import EnrollPage from '@/pages/EnrollPage'
import FaceDetailPage from '@/pages/FaceDetailPage'
import IdentifyPage from '@/pages/IdentifyPage'
import NotFoundPage from '@/pages/NotFoundPage'
import ProcessDetailPage from '@/pages/ProcessDetailPage'
import VideoPage from '@/pages/VideoPage'
import { Route, Routes } from 'react-router'

function AppContent() {
  const { toasts, removeToast } = useToast()

  return (
    <Layout>
      <Routes>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/identify" element={<IdentifyPage />} />
        <Route path="/enroll/:faceId" element={<EnrollPage />} />
        <Route path="/faces/:faceId" element={<FaceDetailPage />} />
        <Route path="/processes/:processId" element={<ProcessDetailPage />} />
        <Route path="/videos" element={<VideoPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
      <ToastContainer toasts={toasts} onRemove={removeToast} />
    </Layout>
  )
}

export default function App() {
  return (
    <ErrorBoundary>
      <AppContent />
    </ErrorBoundary>
  )
}
