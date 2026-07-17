import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import { Shell } from "./layout/Shell";
import { ABTests } from "./pages/ABTests";
import { Analytics } from "./pages/Analytics";
import { AnnotateDetail } from "./pages/AnnotateDetail";
import { AnnotateQueue } from "./pages/AnnotateQueue";
import { Benchmarks } from "./pages/Benchmarks";
import { OpsHealth } from "./pages/OpsHealth";
import { TraceDetail } from "./pages/TraceDetail";
import { TraceList } from "./pages/TraceList";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Shell />}>
          <Route index element={<Navigate to="/traces" replace />} />
          <Route path="traces" element={<TraceList />} />
          <Route path="traces/:traceId" element={<TraceDetail />} />
          <Route path="annotate" element={<AnnotateQueue />} />
          <Route path="annotate/:id" element={<AnnotateDetail />} />
          <Route path="ab" element={<ABTests />} />
          <Route path="analytics" element={<Analytics />} />
          <Route path="benchmarks" element={<Benchmarks />} />
          <Route path="ops" element={<OpsHealth />} />
          <Route path="*" element={<Navigate to="/traces" replace />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
