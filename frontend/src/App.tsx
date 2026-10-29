import { BrowserRouter, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import AlertsPage from "./pages/AlertsPage";
import CasePage from "./pages/CasePage";
import EvaluationPage from "./pages/EvaluationPage";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route index element={<AlertsPage />} />
          <Route path="case/:alertId" element={<CasePage />} />
          <Route path="evaluation" element={<EvaluationPage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
