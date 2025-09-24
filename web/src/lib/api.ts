export type SentimentScores = { positive: number; neutral: number; negative: number };
export type SentimentResponse = { label: 'positive' | 'neutral' | 'negative'; scores: SentimentScores };

const API_BASE = (import.meta.env.VITE_API_BASE as string) || 'http://127.0.0.1:7860';

export async function predictSentiment(text: string): Promise<SentimentResponse> {
  const res = await fetch(`${API_BASE}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    throw new Error(`API error: ${res.status}`);
  }
  const data = await res.json();
  // Map whatever comes to our shape (expecting already in correct shape from FastAPI)
  const label = data.label as 'positive' | 'neutral' | 'negative';
  const scores = data.scores as SentimentScores;
  return { label, scores };
}

