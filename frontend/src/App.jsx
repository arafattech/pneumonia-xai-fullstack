// src/App.jsx

import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Navbar  from './components/Navbar'
import Home    from './pages/Home'
import Explain from './pages/Explain'
import About   from './pages/About'

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/"        element={<Home />} />
        <Route path="/explain" element={<Explain />} />
        <Route path="/about"   element={<About />} />
      </Routes>
    </BrowserRouter>
  )
}
