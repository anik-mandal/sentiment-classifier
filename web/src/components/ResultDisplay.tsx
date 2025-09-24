import ProbabilityChart from './ProbabilityChart'

type Scores = { positive: number; neutral: number; negative: number }

export default function ResultDisplay({ label, scores }: { label: 'positive'|'neutral'|'negative'|''; scores: Scores | null }) {
  const color = label === 'positive' ? 'text-positive' : label === 'negative' ? 'text-negative' : 'text-neutral'
  return (
    <div className="flex flex-col items-center gap-6">
      <div className={`text-5xl md:text-7xl font-extrabold ${color}`}>{label ? label.toUpperCase() : ''}</div>
      {scores && <ProbabilityChart scores={scores} />}
      <div className="text-sm text-gray-400">Thank you. ❤️</div>
    </div>
  )
}

