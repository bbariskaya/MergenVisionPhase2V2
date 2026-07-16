import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Layout from './components/Layout'
import Dashboard from './pages/Dashboard'
import ImageTest from './pages/ImageTest'
import VideoTest from './pages/VideoTest'
import Faces from './pages/Faces'
import Processes from './pages/Processes'
import Analytics from './pages/Analytics'

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Layout />}>
          <Route index element={<Dashboard />} />
          <Route path="image-test" element={<ImageTest />} />
          <Route path="video-test" element={<VideoTest />} />
          <Route path="faces" element={<Faces />} />
          <Route path="processes" element={<Processes />} />
          <Route path="analytics" element={<Analytics />} />
        </Route>
      </Routes>
    </BrowserRouter>
  )
}

export default App
