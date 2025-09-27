import { useMemo } from "react";
import { motion, useReducedMotion } from "framer-motion";
import clsx from "clsx";
import PrimaryButton from "./PrimaryButton";
import { buildRemarksCopy } from "../lib/remarks";
import type {
  PredictionMode,
  PredictionResponse,
  SentimentLabel,
  TopToken,
} from "../lib/api";

type ResultCardProps = {
  prediction: PredictionResponse;
  mode: PredictionMode;
  onModeChange: (mode: PredictionMode) => void;
  onReset: () => void;
  classOrder: SentimentLabel[];
  modeDisabled?: boolean;
};

const MODE_LABELS: Record<PredictionMode, string> = {
  balanced: "Balanced",
  high_precision_negative: "High-precision (NEG)",
};

const MODE_DESCRIPTION = "Balanced thresholds vs stricter NEG thresholds.";

const modeOptions: PredictionMode[] = ["balanced", "high_precision_negative"];

const labelAccent: Record<SentimentLabel, string> = {
  negative: "text-[#EF4444] bg-[#EF4444]/10",
  neutral: "text-[#F59E0B] bg-[#F59E0B]/10",
  positive: "text-[#22C55E] bg-[#22C55E]/10",
};

const barAccent: Record<SentimentLabel, string> = {
  negative: "bg-gradient-to-r from-[#EF4444] to-[#7F1D1D]",
  neutral: "bg-gradient-to-r from-[#F59E0B] to-[#92400E]",
  positive: "bg-gradient-to-r from-[#22C55E] to-[#14532D]",
};

const MotionDiv = motion.div;

const TopTokenChip = ({ token, weight }: TopToken) => {
  const formatted = useMemo(() => `${token}`.trim(), [token]);
  return (
    <span
      className="rounded-full border border-[#1F2937] bg-[#111827] px-3 py-1 text-xs text-[#E5E7EB]/80 shadow-sm"
      title={`${formatted} (weight ${weight.toFixed(3)})`}
    >
      {formatted.length > 18 ? `${formatted.slice(0, 18)}...` : formatted}
    </span>
  );
};

const ConfidenceBar = ({
  label,
  value,
  reduced,
}: {
  label: string;
  value: number;
  reduced: boolean;
}) => {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center justify-between text-xs uppercase tracking-widest text-[#9CA3AF]">
        <span>{label}</span>
        <span className="font-semibold text-[#E5E7EB]">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-3 w-full overflow-hidden rounded-full bg-[#1F2937]">
        <MotionDiv
          className="h-full rounded-full bg-[#06B6D4]"
          style={{ originX: 0 }}
          initial={{ width: reduced ? `${pct}%` : "0%" }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.35, ease: "easeOut" }}
        />
      </div>
    </div>
  );
};

const ClassBar = ({
  label,
  value,
  accent,
  reduced,
}: {
  label: SentimentLabel;
  value: number;
  accent: string;
  reduced: boolean;
}) => {
  const pct = Math.max(0, Math.min(100, value * 100));
  return (
    <div className="flex items-center gap-3">
      <span className="min-w-[72px] text-sm font-medium text-[#9CA3AF] capitalize">{label}</span>
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-[#1F2937]">
        <MotionDiv
          className={clsx("absolute inset-y-0 left-0", accent)}
          style={{ originX: 0 }}
          initial={{ width: reduced ? `${pct}%` : "0%" }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.35, ease: "easeOut" }}
        />
      </div>
      <span className="w-14 text-right text-sm text-[#E5E7EB]">{pct.toFixed(1)}%</span>
    </div>
  );
};

const ResultCard = ({ prediction, mode, onModeChange, onReset, classOrder, modeDisabled = false }: ResultCardProps) => {
  const reducedMotion = useReducedMotion();
  const { label, confidence, confidences, probabilities, top_tokens, remarks } = prediction;
  const copyText = remarks.copy_text || buildRemarksCopy(prediction);

  return (
    <MotionDiv
      key={`${label}-${mode}`}
      initial={reducedMotion ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.22, ease: "easeOut" }}
      className="w-full max-w-4xl rounded-[12px] border border-[#1F2937] bg-[#0B1220] p-8 text-left shadow-xl"
    >
      <div className="flex flex-col gap-6">
        <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
          <MotionDiv
            initial={reducedMotion ? { scale: 1 } : { scale: 0.95, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ duration: 0.12, ease: "easeOut" }}
            className={clsx(
              "inline-flex items-center gap-4 rounded-full px-6 py-3 text-lg font-semibold",
              labelAccent[label],
            )}
            aria-label={`Predicted label ${label}`}
          >
            <span className="text-sm uppercase tracking-[0.32em] text-[#9CA3AF]">Label</span>
            <span className="text-2xl text-[#E5E7EB] capitalize">{label}</span>
          </MotionDiv>

          <div className="flex items-center gap-2 rounded-full border border-[#1F2937] bg-[#111827] p-1 text-sm text-[#E5E7EB]">
            {modeOptions.map((option) => (
              <button
                key={option}
                type="button"
                onClick={() => {
                  if (modeDisabled || option === mode) return;
                  onModeChange(option);
                }}
                disabled={modeDisabled}
                className={clsx(
                  "rounded-full px-4 py-2 transition focus:outline-none focus-visible:ring-2 focus-visible:ring-[#06B6D4]",
                  option === mode ? "bg-[#1F2937] text-[#E5E7EB]" : "text-[#9CA3AF]",
                  modeDisabled && option !== mode ? "cursor-not-allowed opacity-60" : "",
                )}
              >
                {MODE_LABELS[option]}
              </button>
            ))}
          </div>
        </div>

        <div className="rounded-[12px] border border-[#1F2937] bg-[#111827] p-6 text-[#E5E7EB] shadow">
          <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-[#9CA3AF]">Calibrated Confidence</h3>
          <div className="mt-4">
            <ConfidenceBar label="Predicted" value={confidence} reduced={Boolean(reducedMotion)} />
          </div>
          <p className="mt-3 text-xs text-[#9CA3AF]">{remarks.caption ?? MODE_DESCRIPTION}</p>
        </div>

        <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
          <div className="rounded-[12px] border border-[#1F2937] bg-[#111827] p-6">
            <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-[#9CA3AF]">Per-class Confidence</h3>
            <div className="mt-4 flex flex-col gap-3">
              {classOrder.map((cls) => (
                <ClassBar
                  key={cls}
                  label={cls}
                  value={confidences[cls] ?? 0}
                  accent={barAccent[cls]}
                  reduced={Boolean(reducedMotion)}
                />
              ))}
            </div>
          </div>

          <div className="rounded-[12px] border border-[#1F2937] bg-[#111827] p-6">
            <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-[#9CA3AF]">Model Probabilities</h3>
            <div className="mt-4 flex flex-col gap-3">
              {classOrder.map((cls) => (
                <div key={cls} className="flex items-center justify-between text-sm text-[#E5E7EB]">
                  <span className="capitalize">{cls}</span>
                  <span>{(probabilities[cls] * 100).toFixed(1)}%</span>
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="rounded-[12px] border border-[#1F2937] bg-[#111827] p-6 text-[#E5E7EB]">
          <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-[#9CA3AF]">Top tokens / phrases</h3>
          <div className="mt-4 flex flex-wrap gap-2">
            {top_tokens.length ? (
              top_tokens.map((token) => <TopTokenChip key={token.token} {...token} />)
            ) : (
              <span className="text-sm text-[#9CA3AF]">No salient tokens available.</span>
            )}
          </div>
        </div>

        <div className="rounded-[12px] border border-[#1F2937] bg-[#111827] p-6 text-[#E5E7EB]">
          <div className="flex flex-col gap-4 md:flex-row md:items-start md:justify-between">
            <div className="flex-1 space-y-3">
              <h3 className="text-sm font-semibold uppercase tracking-[0.24em] text-[#9CA3AF]">Remarks (auto)</h3>
              <p className="text-sm text-[#E5E7EB]">{remarks.summary}</p>
              <p className="text-sm text-[#E5E7EB]">{remarks.rationale}</p>
              {remarks.doc && <p className="text-sm text-[#E5E7EB]">{remarks.doc}</p>}
              <p className="text-sm font-medium text-[#06B6D4]">{remarks.next_action}</p>
            </div>
            <button
              type="button"
              onClick={() => {
                void navigator.clipboard?.writeText(copyText);
              }}
              className="mt-4 inline-flex items-center rounded-full border border-[#06B6D4] px-4 py-2 text-sm font-medium text-[#06B6D4] transition hover:bg-[#06B6D4]/10 focus:outline-none focus-visible:ring-2 focus-visible:ring-[#06B6D4] md:mt-0"
            >
              Copy remarks
            </button>
          </div>
        </div>

        <div className="flex flex-col items-center justify-between gap-4 text-center md:flex-row md:text-left">
          <p className="text-xs uppercase tracking-[0.32em] text-[#9CA3AF]">Confidence is temperature-calibrated; thresholds per mode.</p>
          <PrimaryButton onClick={onReset}>Run another</PrimaryButton>
        </div>
      </div>
    </MotionDiv>
  );
};

export default ResultCard;
