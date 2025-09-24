import { ResponsiveContainer, BarChart, Bar, XAxis, YAxis, Tooltip, LabelList } from 'recharts'

type Scores = { positive: number; neutral: number; negative: number }

export default function ProbabilityChart({ scores }: { scores: Scores }) {
  const data = [
    { name: 'Positive', value: Math.round(scores.positive * 100), color: '#22c55e' },
    { name: 'Neutral', value: Math.round(scores.neutral * 100), color: '#6b7280' },
    { name: 'Negative', value: Math.round(scores.negative * 100), color: '#ef4444' },
  ]
  return (
    <div className="w-full h-64 glass">
      <ResponsiveContainer width="100%" height="100%">
        <BarChart data={data} margin={{ top: 20, right: 20, bottom: 0, left: 20 }}>
          <XAxis dataKey="name" stroke="#9ca3af" tickLine={false} axisLine={false} />
          <YAxis hide domain={[0, 100]} />
          <Tooltip formatter={(v: any) => `${v}%`} contentStyle={{ background: '#111827', border: '1px solid #1f2937' }} />
          <Bar dataKey="value" radius={[6,6,0,0]}>
            <LabelList dataKey="value" position="top" formatter={(v: any) => `${v}%`} fill="#e5e7eb" />
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  )
}

