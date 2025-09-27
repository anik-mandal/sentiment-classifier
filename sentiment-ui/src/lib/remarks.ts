import type { PredictionResponse } from "./api";

export const buildRemarksCopy = (prediction: PredictionResponse): string => {
  const { label, confidence, top_tokens, remarks } = prediction;
  const summary = remarks?.summary ?? `This comment is likely ${label}.`;
  const confText = `Calibrated confidence ${(confidence * 100).toFixed(1)}%.`;
  const tokenSummary = top_tokens?.length
    ? `Driven by tokens: ${top_tokens.slice(0, 2).map((t) => t.token).join(", ")}.`
    : "Key tokens unavailable.";
  const nextAction = remarks?.next_action ?? "Next action: Monitor.";
  return [summary, confText, tokenSummary, remarks?.doc, nextAction]
    .filter(Boolean)
    .join("\n");
};
