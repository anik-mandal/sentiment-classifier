import { useCallback, useEffect, useMemo, useState } from "react";
import { AnimatePresence } from "framer-motion";
import Section from "./components/Section";
import PrimaryButton from "./components/PrimaryButton";
import InputModeCard from "./components/InputModeCard";
import ResultCard from "./components/ResultCard";
import {
  fetchOperatingPoint,
  predictBatch,
  predictText,
  type OperatingPointConfig,
  type PredictionMode,
  type PredictionResponse,
  type SentimentLabel,
} from "./lib/api";
import { extractCsvRows, type CsvParseResult } from "./lib/csv";
import { extractPdfText, parsePageRange } from "./lib/pdf";

type InputMode = "text" | "csv" | "pdf";
type Step = 1 | 2 | 3;

type PdfState = {
  file: File;
  fileName: string;
  totalPages: number;
  pages?: number[];
  rangeType: "all" | "range";
  rangeValue: string;
  text: string;
};

type CsvState = CsvParseResult & {
  fileName: string;
};

type LastRequest =
  | { kind: "text"; text: string; meta?: Record<string, unknown> }
  | { kind: "csv"; rows: string[]; meta?: Record<string, unknown> }
  | { kind: "pdf"; text: string; meta?: Record<string, unknown> };

const INPUT_MODES: InputMode[] = ["text", "csv", "pdf"];

const modeCopy: Record<InputMode, { title: string; description: string; icon: string }> = {
  text: {
    title: "Text",
    description: "Paste or type freeform content. Instant single-sample feedback.",
    icon: "TXT",
  },
  csv: {
    title: "CSV",
    description: "Batch predict rows from spreadsheets. Scores are averaged for you.",
    icon: "CSV",
  },
  pdf: {
    title: "PDF",
    description: "Extract pages locally, choose ranges, and analyse rich documents.",
    icon: "PDF",
  },
};

const DEFAULT_CLASS_ORDER: SentimentLabel[] = ["negative", "neutral", "positive"];

const App = () => {
  const [currentStep, setCurrentStep] = useState<Step>(1);
  const [direction, setDirection] = useState(1);
  const [mode, setMode] = useState<InputMode>("text");
  const [text, setText] = useState("");
  const [csvState, setCsvState] = useState<CsvState | null>(null);
  const [pdfState, setPdfState] = useState<PdfState | null>(null);
  const [loading, setLoading] = useState(false);
  const [inputBusy, setInputBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [activeMode, setActiveMode] = useState<PredictionMode>("balanced");
  const [operatingPoint, setOperatingPoint] = useState<OperatingPointConfig | null>(null);
  const [classOrder, setClassOrder] = useState<SentimentLabel[]>(DEFAULT_CLASS_ORDER);
  const [lastRequest, setLastRequest] = useState<LastRequest | null>(null);

  useEffect(() => {
    let cancelled = false;
    const loadConfig = async () => {
      try {
        const config = await fetchOperatingPoint();
        if (cancelled) return;
        setOperatingPoint(config);
        if (Array.isArray(config.class_order) && config.class_order.length) {
          setClassOrder(config.class_order as SentimentLabel[]);
        }
      } catch (err) {
        console.warn("Failed to load operating point", err);
      }
    };
    loadConfig();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!operatingPoint || !import.meta.env.DEV) return;
    let run = true;
    const runAssertions = async () => {
      try {
        const neg = await predictText("This service is terrible and unacceptable.", "balanced", { source: "dev" });
        if (run) {
          console.assert(neg.label === "negative" && neg.confidence > 0.6, "Expected negative high confidence");
        }
        const borderline = "Slight delay but issue resolved eventually.";
        const neutralBalanced = await predictText(borderline, "balanced", { source: "dev" });
        const neutralStrict = await predictText(borderline, "high_precision_negative", { source: "dev" });
        if (run) {
          console.assert(
            neutralStrict.label !== "negative",
            "High precision mode should avoid flipping borderline neutral to negative",
          );
        }
        const positive = await predictText("Great support experience, very helpful staff!", "balanced", { source: "dev" });
        if (run) {
          console.assert(
            positive.label === "positive" && positive.probabilities.positive >= 0.8,
            "Positive sample should have strong probability",
          );
        }
      } catch (err) {
        if (run) {
          console.warn("Dev assertions could not run", err);
        }
      }
    };
    runAssertions();
    return () => {
      run = false;
    };
  }, [operatingPoint]);

  const goToStep = useCallback(
    (next: Step) => {
      setDirection(next > currentStep ? 1 : -1);
      setCurrentStep(next);
    },
    [currentStep],
  );

  const handleReset = () => {
    setResult(null);
    setError(null);
    setLoading(false);
    setCsvState(null);
    setPdfState(null);
    setText("");
    setDirection(-1);
    setCurrentStep(2);
    setLastRequest(null);
  };

  const handleSelectMode = (value: InputMode) => {
    setMode(value);
    setError(null);
  };

  const handleCsvFile = async (fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;
    setInputBusy(true);
    setError(null);
    try {
      const parsed = await extractCsvRows(file);
      setCsvState({ ...parsed, fileName: file.name });
    } catch (err) {
      setCsvState(null);
      setError(err instanceof Error ? err.message : "Failed to process CSV file.");
    } finally {
      setInputBusy(false);
    }
  };

  const handlePdfFile = async (fileList: FileList | null) => {
    const file = fileList?.[0];
    if (!file) return;
    setInputBusy(true);
    setError(null);
    try {
      const { text: pdfText, totalPages } = await extractPdfText(file);
      setPdfState({
        file,
        fileName: file.name,
        totalPages,
        pages: undefined,
        rangeType: "all",
        rangeValue: "",
        text: pdfText,
      });
    } catch (err) {
      setPdfState(null);
      setError(err instanceof Error ? err.message : "Failed to read PDF content.");
    } finally {
      setInputBusy(false);
    }
  };

  const handlePdfRangeChange = async (range: string, type: "all" | "range") => {
    if (!pdfState) return;
    setPdfState((prev) => (prev ? { ...prev, rangeType: type, rangeValue: range } : prev));

    if (type === "all") {
      setInputBusy(true);
      try {
        const { text: pdfText } = await extractPdfText(pdfState.file);
        setPdfState((prev) =>
          prev
            ? {
                ...prev,
                text: pdfText,
                pages: undefined,
              }
            : prev,
        );
      } catch (err) {
        setError(err instanceof Error ? err.message : "Failed to refresh PDF content.");
      } finally {
        setInputBusy(false);
      }
      return;
    }

    if (!range.trim()) {
      setPdfState((prev) => (prev ? { ...prev, pages: [], text: "" } : prev));
      return;
    }

    setInputBusy(true);
    try {
      const pages = parsePageRange(range, pdfState.totalPages);
      const { text: pdfText } = await extractPdfText(pdfState.file, pages);
      setPdfState((prev) =>
        prev
          ? {
              ...prev,
              pages,
              text: pdfText,
            }
          : prev,
      );
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "The page range is invalid.");
    } finally {
      setInputBusy(false);
    }
  };

  const canPredict = useMemo(() => {
    if (mode === "text") {
      return text.trim().length > 0;
    }
    if (mode === "csv") {
      return Boolean(csvState?.rows?.length);
    }
    if (mode === "pdf") {
      return Boolean(pdfState?.text?.length);
    }
    return false;
  }, [mode, text, csvState, pdfState]);

  const runPrediction = useCallback(
    async (targetMode: PredictionMode, request: LastRequest) => {
      setLoading(true);
      setError(null);
      try {
        let prediction: PredictionResponse;
        if (request.kind === "text") {
          prediction = await predictText(request.text, targetMode, request.meta);
        } else if (request.kind === "csv") {
          prediction = await predictBatch(request.rows, targetMode, request.meta);
        } else {
          prediction = await predictText(request.text, targetMode, request.meta);
        }
        setResult(prediction);
        setActiveMode(targetMode);
        setDirection(1);
        setCurrentStep(3);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Prediction failed. Please try again.");
      } finally {
        setLoading(false);
      }
    },
    [],
  );

  const handlePredict = async () => {
    if (!canPredict || loading) return;
    if (mode === "text") {
      const payload: LastRequest = {
        kind: "text",
        text: text.trim(),
        meta: { source: "text", length: text.trim().length },
      };
      setLastRequest(payload);
      await runPrediction(activeMode, payload);
      return;
    }
    if (mode === "csv" && csvState?.rows?.length) {
      const payload: LastRequest = {
        kind: "csv",
        rows: csvState.rows,
        meta: { source: "csv", count: csvState.rows.length, file: csvState.fileName },
      };
      setLastRequest(payload);
      await runPrediction(activeMode, payload);
      return;
    }
    if (mode === "pdf" && pdfState?.text) {
      const payload: LastRequest = {
        kind: "pdf",
        text: pdfState.text,
        meta: {
          source: "pdf",
          pages: pdfState.pages?.join(", ") ?? "all",
          file: pdfState.fileName,
        },
      };
      setLastRequest(payload);
      await runPrediction(activeMode, payload);
      return;
    }
  };

  const handleModeToggle = (nextMode: PredictionMode) => {
    if (nextMode === activeMode || loading) return;
    if (lastRequest) {
      void runPrediction(nextMode, lastRequest);
    } else {
      setActiveMode(nextMode);
    }
  };

  const heroSection = (
    <div className="flex w-full max-w-5xl flex-col items-center gap-10 text-center">
      <div className="hero-gradient glass-card fade-border relative flex flex-col items-center gap-6 overflow-hidden px-10 py-16">
        <span className="text-xs uppercase tracking-[0.65rem] text-white/50">Multilingual Sentiment</span>
        <h1 className="text-glow text-5xl font-semibold leading-tight text-white md:text-6xl">
          Multilingual Sentiment (XLM-R)
        </h1>
        <p className="max-w-2xl text-lg text-white/70">
          Fast, local, GPU-ready sentiment across Bangla, Hinglish, and English.
        </p>
        <ul className="mt-6 grid w-full gap-3 text-left text-base text-white/70 md:grid-cols-2">
          <li className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">Predicts positive / neutral / negative for Text, CSV rows, or PDF content.</li>
          <li className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">Works with my local backend (Gradio/Flask compatible).</li>
          <li className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">Returns label, calibrated confidence, and per-class insights.</li>
          <li className="rounded-xl border border-white/10 bg-white/5 px-4 py-3">Designed for offline/localhost demos.</li>
        </ul>
        <div className="mt-8">
          <PrimaryButton onClick={() => goToStep(2)}>Let's Go →</PrimaryButton>
        </div>
      </div>
    </div>
  );

  const inputSection = (
    <div className="flex w-full max-w-6xl flex-col gap-10">
      <div className="flex flex-col items-start gap-4 text-left">
        <span className="text-xs uppercase tracking-[0.6rem] text-white/50">Input</span>
        <h2 className="text-4xl font-semibold text-white">Select your source</h2>
        <p className="max-w-2xl text-base text-white/70">
          Choose how you want to analyse sentiment. All parsing happens locally in the browser before
          hitting your localhost endpoint.
        </p>
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {INPUT_MODES.map((item) => (
          <InputModeCard
            key={item}
            mode={item}
            title={modeCopy[item].title}
            description={modeCopy[item].description}
            icon={modeCopy[item].icon}
            active={mode === item}
            onSelect={handleSelectMode}
          />
        ))}
      </div>

      <div className="glass-card fade-border relative min-h-[320px] w-full overflow-hidden p-8">
        {mode === "text" && (
          <div className="flex h-full flex-col gap-6">
            <label htmlFor="text-input" className="text-sm uppercase tracking-[0.4rem] text-white/50">
              Text sample
            </label>
            <textarea
              id="text-input"
              placeholder="Type or paste text in Bangla, Hinglish, or English..."
              className="h-48 w-full resize-none rounded-2xl border border-white/10 bg-white/5 px-4 py-3 text-base text-white/80 outline-none transition focus:border-accent/60 focus:bg-white/10"
              value={text}
              onChange={(event) => setText(event.target.value)}
            />
            <p className="text-sm text-white/40">Badge legend: red = negative, gray = neutral, green = positive.</p>
          </div>
        )}

        {mode === "csv" && (
          <div className="flex h-full flex-col gap-4">
            <label className="text-sm uppercase tracking-[0.4rem] text-white/50">Upload CSV</label>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-white/20 bg-white/5 px-6 py-10 text-center text-white/70 transition hover:border-white/40 hover:bg-white/10">
              <input
                type="file"
                accept=".csv"
                className="hidden"
                onChange={(event) => {
                  void handleCsvFile(event.target.files);
                  event.target.value = "";
                }}
              />
              <span className="text-lg font-medium">Drop or choose a .csv file</span>
              <span className="text-sm text-white/50">We average sentiment across every parsed row.</span>
            </label>
            {csvState && (
              <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-sm text-white/60">
                <div className="flex items-center justify-between text-white/70">
                  <span className="font-medium">{csvState.fileName}</span>
                  <span>{csvState.rows.length} rows</span>
                </div>
                <div className="mt-3 space-y-2">
                  {csvState.preview.map((item, index) => (
                    <p key={index} className="truncate text-xs text-white/50">
                      {item}
                    </p>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {mode === "pdf" && (
          <div className="flex h-full flex-col gap-5">
            <label className="text-sm uppercase tracking-[0.4rem] text-white/50">Upload PDF</label>
            <label className="flex cursor-pointer flex-col items-center justify-center gap-3 rounded-2xl border border-dashed border-white/20 bg-white/5 px-6 py-10 text-center text-white/70 transition hover:border-white/40 hover:bg-white/10">
              <input
                type="file"
                accept="application/pdf"
                className="hidden"
                onChange={(event) => {
                  void handlePdfFile(event.target.files);
                  event.target.value = "";
                }}
              />
              <span className="text-lg font-medium">Drop or choose a .pdf file</span>
              <span className="text-sm text-white/50">Content is extracted locally using pdf.js.</span>
            </label>

            {pdfState && (
              <div className="space-y-4 text-sm text-white/70">
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <span className="font-medium">{pdfState.fileName}</span>
                  <span className="text-white/50">{pdfState.totalPages} pages available</span>
                </div>
                <div className="flex flex-col gap-3 md:flex-row md:items-center">
                  <div className="flex items-center gap-2">
                    <input
                      type="radio"
                      id="pdf-all"
                      name="pdf-range"
                      className="h-4 w-4 accent-accent"
                      checked={pdfState.rangeType === "all"}
                      onChange={() => void handlePdfRangeChange("", "all")}
                    />
                    <label htmlFor="pdf-all" className="text-white/60">
                      All pages
                    </label>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      type="radio"
                      id="pdf-range"
                      name="pdf-range"
                      className="h-4 w-4 accent-accent"
                      checked={pdfState.rangeType === "range"}
                      onChange={() => void handlePdfRangeChange(pdfState.rangeValue || "1", "range")}
                    />
                    <label htmlFor="pdf-range" className="text-white/60">
                      Page range
                    </label>
                    <input
                      type="text"
                      placeholder="e.g. 1-3,5"
                      value={pdfState.rangeValue}
                      onChange={(event) => void handlePdfRangeChange(event.target.value, "range")}
                      className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-sm text-white/80 outline-none transition focus:border-accent/60 focus:bg-white/10"
                    />
                  </div>
                </div>
                <div className="rounded-2xl border border-white/10 bg-white/5 p-4 text-xs text-white/50">
                  {pdfState.text ? (
                    <div className="max-h-48 overflow-y-auto whitespace-pre-wrap text-white/60">
                      {pdfState.text}
                    </div>
                  ) : (
                    <p>Select a page range to preview the extracted text.</p>
                  )}
                </div>
              </div>
            )}
          </div>
        )}
        {inputBusy && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-bg/60">
            <div className="h-16 w-16 animate-spin rounded-full border-2 border-white/30 border-t-accent" />
          </div>
        )}
      </div>

      <div className="flex flex-col items-start gap-4 md:flex-row md:items-center md:justify-between">
        <div className="text-sm text-white/50">Works with your local POST endpoint at {import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:7860"}.</div>
        <div className="flex flex-col items-end gap-2">
          <PrimaryButton onClick={handlePredict} disabled={!canPredict || loading} loading={loading}>
            Predict
          </PrimaryButton>
          {error && <span className="text-sm text-negative">{error}</span>}
        </div>
      </div>
    </div>
  );

  const resultsSection = result ? (
    <ResultCard
      prediction={result}
      mode={activeMode}
      onModeChange={handleModeToggle}
      onReset={handleReset}
      classOrder={classOrder}
      modeDisabled={loading}
    />
  ) : (
    <div className="glass-card fade-border flex h-72 w-full max-w-3xl flex-col items-center justify-center gap-4">
      <div className="h-6 w-44 animate-pulse rounded-full bg-white/10" />
      <div className="h-56 w-full max-w-md animate-pulse rounded-2xl bg-white/5" />
      <div className="h-10 w-32 animate-pulse rounded-full bg-white/10" />
    </div>
  );

  const sections: Record<Step, JSX.Element> = {
    1: heroSection,
    2: inputSection,
    3: resultsSection,
  };

  return (
    <div className="layer-gradient relative flex h-screen w-screen items-center justify-center overflow-hidden">
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(circle_at_top,_rgba(127,90,240,0.4),_transparent_55%)]" />
      <AnimatePresence initial={false} custom={direction} mode="wait">
        <Section key={currentStep} id={currentStep} direction={direction}>
          {sections[currentStep]}
        </Section>
      </AnimatePresence>
    </div>
  );
};

export default App;
