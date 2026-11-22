import { BrowserRouter, Route, Routes } from "react-router-dom";
import { Shell } from "./components/Shell";
import { ProgressProvider } from "./lib/progress";
import { CertificatePage } from "./pages/CertificatePage";
import { DashboardPage } from "./pages/DashboardPage";
import { HomePage } from "./pages/HomePage";
import { LessonPage } from "./pages/LessonPage";
import { ModulePage } from "./pages/ModulePage";
import { NotFoundPage } from "./pages/NotFoundPage";

const basename = import.meta.env.BASE_URL.replace(/\/$/, "");

export default function App() {
  return (
    <ProgressProvider>
      <BrowserRouter basename={basename}>
        <Routes>
          <Route element={<Shell />}>
            <Route index element={<HomePage />} />
            <Route path="module/:moduleId" element={<ModulePage />} />
            <Route path="lesson/:lessonId" element={<LessonPage />} />
            <Route path="dashboard" element={<DashboardPage />} />
            <Route path="certificate" element={<CertificatePage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </ProgressProvider>
  );
}
