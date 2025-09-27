export type SentimentLabel = "negative" | "neutral" | "positive";

export type SentimentScores = Record<SentimentLabel, number>;

export type TopClass = {
  label: SentimentLabel;
  confidence: number;
};

export type TopToken = {
  token: string;
  weight: number;
};

export type Remarks = {
  summary: string;
  rationale: string;
  doc?: string;
  next_action: string;
  caption?: string;
  copy_text: string;
};

export type PredictionMode = "balanced" | "high_precision_negative";

export type PredictionResponse = {
  label: SentimentLabel;
  confidence: number;
  confidences: SentimentScores;
  probabilities: SentimentScores;
  scores: SentimentScores;
  top_classes: TopClass[];
  top_tokens: TopToken[];
  mode: PredictionMode;
  remarks: Remarks;
  operating_point: {
    class_order: SentimentLabel[];
    mode: PredictionMode;
  };
  counts?: Record<string, number>;
};

export type OperatingPointConfig = {
  class_order: SentimentLabel[];
  modes: Record<string, unknown>;
  copy?: {
    caption?: string;
    next_actions?: Record<SentimentLabel, string>;
  };
};

const API_BASE = import.meta.env.VITE_API_BASE ?? "http://127.0.0.1:7860";

const orderedLabels: SentimentLabel[] = ["negative", "neutral", "positive"];

const normalizeScores = (scores: Partial<SentimentScores>): SentimentScores => {
  const fallback: SentimentScores = { negative: 0.33, neutral: 0.34, positive: 0.33 };
  const entries = orderedLabels.map((label) => [label, scores[label] ?? 0]);
  const total = entries.reduce((sum, [, value]) => sum + value, 0);
  if (total <= 0) {
    return fallback;
  }
  return Object.fromEntries(
    entries.map(([label, value]) => [label, Number((value / total).toFixed(6))]),
  ) as SentimentScores;
};

export const fetchOperatingPoint = async (): Promise<OperatingPointConfig> => {
  const response = await fetch(`${API_BASE}/operating-point`);
  if (!response.ok) {
    throw new Error(`Failed to load operating point (${response.status})`);
  }
  const json = (await response.json()) as OperatingPointConfig;
  return json;
};

export const predictText = async (
  text: string,
  mode: PredictionMode,
  meta?: Record<string, unknown>,
): Promise<PredictionResponse> => {
  const response = await fetch(`${API_BASE}/predict`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ text, mode, meta }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Request failed with status ${response.status}`);
  }

  const payload = (await response.json()) as PredictionResponse;
  payload.confidences = normalizeScores(payload.confidences ?? {});
  payload.probabilities = normalizeScores(payload.probabilities ?? {});
  payload.scores = normalizeScores(payload.scores ?? {});
  return payload;
};

export const predictBatch = async (
  texts: string[],
  mode: PredictionMode,
  meta?: Record<string, unknown>,
): Promise<PredictionResponse> => {
  const response = await fetch(`${API_BASE}/predict/batch`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ texts, mode, meta }),
  });

  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `Batch request failed with status ${response.status}`);
  }

  const payload = (await response.json()) as PredictionResponse;
  payload.confidences = normalizeScores(payload.confidences ?? {});
  payload.probabilities = normalizeScores(payload.probabilities ?? {});
  payload.scores = normalizeScores(payload.scores ?? {});
  return payload;
};

export const averageScoreSets = (scoreSets: SentimentScores[]): SentimentScores => {
  if (!scoreSets.length) {
    return { negative: 0, neutral: 0, positive: 0 };
  }
  const totals = scoreSets.reduce<SentimentScores>(
    (acc, scores) => ({
      negative: acc.negative + (scores.negative ?? 0),
      neutral: acc.neutral + (scores.neutral ?? 0),
      positive: acc.positive + (scores.positive ?? 0),
    }),
    { negative: 0, neutral: 0, positive: 0 },
  );

  const averaged: SentimentScores = {
    negative: totals.negative / scoreSets.length,
    neutral: totals.neutral / scoreSets.length,
    positive: totals.positive / scoreSets.length,
  };
  return normalizeScores(averaged);
};

export const getDominantLabel = (scores: SentimentScores): SentimentLabel => {
  return orderedLabels.reduce((best, label) => {
    return scores[label] > scores[best] ? label : best;
  }, orderedLabels[0]);
};
