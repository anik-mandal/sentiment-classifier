import { useRef, useState } from 'react'
import { motion, useInView } from 'framer-motion'
import Section from '../components/Section'
import PrimaryButton from '../components/PrimaryButton'
import InputModeCard from '../components/InputModeCard'
import ResultDisplay from '../components/ResultDisplay'
import { predictSentiment, SentimentResponse } from '../lib/api'
import { readCsvAsTextBlob } from '../lib/csv'
import { extractPdfText } from '../lib/pdf'

type Mode = 'text' | 'csv' | 'pdf'

export default function App() {
  const s1Ref = useRef<HTMLDivElement>(null)
  const s2Ref = useRef<HTMLDivElement>(null)
  const s3Ref = useRef<HTMLDivElement>(null)

  const [mode, setMode] = useState<Mode>('text')
  const [text, setText] = useState('')
  const [csvInfo, setCsvInfo] = useState<{ count: number; headers: string[] } | null>(null)
  const [pdfInfo, setPdfInfo] = useState<{ pageCount: number } | null>(null)
  const [file, setFile] = useState<File | null>(null)
  const [loading, setLoading] = useState(false)
  const [result, setResult] = useState<SentimentResponse | null>(null)

  const scrollTo = (el: HTMLDivElement | null) => el?.scrollIntoView({ behavior: 'smooth', block: 'start' })

  async function onPredict() {
    try {
      setLoading(true)
      let payload = text
      if (mode === 'csv' && file) {
        const { text: t, count, headers } = await readCsvAsTextBlob(file)
        setCsvInfo({ count, headers })
        payload = t
      }
      if (mode === 'pdf' && file) {
        const { text: t, pageCount } = await extractPdfText(file)
        setPdfInfo({ pageCount })
        payload = t
      }
      const res = await predictSentiment(payload)
      setResult(res)
      scrollTo(s3Ref.current!)
    } catch (e) {
      console.error(e)
    } finally {
      setLoading(false)
    }
  }

  function resetToInput() {
    setResult(null)
    setText('')
    setFile(null)
    scrollTo(s2Ref.current!)
  }

  return (
    <main className="font-inter">
      {/* Section 1: About */}
      <div ref={s1Ref as any} />
      <Section id="About">
        <div className="min-h-[80vh] flex items-center">
          <div className="grid md:grid-cols-2 gap-10 items-center w-full">
            <div>
              <div className="text-5xl md:text-7xl font-extrabold leading-tight bg-clip-text text-transparent bg-gradient-to-b from-emerald-400 to-gray-300">Multilingual Sentiment (XLM-R)</div>
              <p className="mt-4 text-gray-300 text-lg">Fast, local, GPU-ready sentiment analysis for Bangla, Hinglish, and English.</p>
              <ul className="mt-6 space-y-2 text-gray-300">
                <li>• Predicts positive / neutral / negative for text, CSV rows, or PDF content.</li>
                <li>• Works with my existing local model server (Gradio-style API).</li>
                <li>• Outputs class label + confidence scores and a probability chart.</li>
                <li>• Designed for offline or localhost use during demos.</li>
                <li>• Badge key: <span className="text-negative">red=negative</span>, <span className="text-neutral">gray=neutral</span>, <span className="text-positive">green=positive</span>.</li>
              </ul>
              <div className="mt-10">
                <PrimaryButton label="Let's Go →" onClick={() => scrollTo(s2Ref.current!)} />
              </div>
            </div>
            <div className="glass p-6">
              <div className="text-center text-3xl font-bold text-neutral">Preview</div>
              <div className="mt-6 text-center text-5xl font-extrabold text-positive">POSITIVE</div>
              <div className="mt-6 h-2 w-full bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-positive" style={{ width: '72%' }} />
              </div>
              <div className="mt-2 h-2 w-full bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-neutral" style={{ width: '18%' }} />
              </div>
              <div className="mt-2 h-2 w-full bg-white/10 rounded-full overflow-hidden">
                <div className="h-full bg-negative" style={{ width: '10%' }} />
              </div>
            </div>
          </div>
        </div>
      </Section>

      {/* Section 2: Input */}
      <div ref={s2Ref as any} />
      <Section id="Input">
        <div className="mb-6 text-xl text-gray-300">Choose your input type.</div>
        <div className="grid md:grid-cols-3 gap-4">
          <InputModeCard title="Text" active={mode==='text'} onClick={() => setMode('text')}>
            <textarea className="mt-2 w-full h-36 glass p-3 outline-none" placeholder="Type or paste your text..." value={text} onChange={e=>setText(e.target.value)} />
          </InputModeCard>
          <InputModeCard title="CSV" active={mode==='csv'} onClick={() => setMode('csv')}>
            <input type="file" accept=".csv" onChange={e=>setFile(e.target.files?.[0]||null)} className="mt-2" />
            {csvInfo && <div className="text-sm text-gray-400 mt-2">Rows: {csvInfo.count}</div>}
          </InputModeCard>
          <InputModeCard title="PDF" active={mode==='pdf'} onClick={() => setMode('pdf')}>
            <input type="file" accept="application/pdf" onChange={e=>setFile(e.target.files?.[0]||null)} className="mt-2" />
            {pdfInfo && <div className="text-sm text-gray-400 mt-2">Pages: {pdfInfo.pageCount}</div>}
          </InputModeCard>
        </div>
        <div className="mt-8">
          <PrimaryButton label={loading ? 'Predicting…' : 'Predict'} onClick={onPredict} disabled={loading} />
        </div>
      </Section>

      {/* Section 3: Results */}
      <div ref={s3Ref as any} />
      <Section id="Results">
        <div className="flex flex-col items-center gap-6 w-full">
          <div className="text-gray-400">Result</div>
          <div className="w-full max-w-3xl">
            <ResultDisplay label={result?.label ?? ''} scores={result?.scores ?? null} />
          </div>
          <button className="mt-6 text-gray-400 underline" onClick={resetToInput}>Run another</button>
        </div>
      </Section>
    </main>
  )
}

