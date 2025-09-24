import Papa from 'papaparse'

export async function readCsvAsTextBlob(file: File): Promise<{ text: string; count: number; headers: string[] }>{
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      header: true,
      skipEmptyLines: true,
      complete: (results) => {
        const rows = results.data as Record<string, string>[]
        const headers = results.meta.fields || []
        let col = headers.find(h => h.toLowerCase() === 'text') || headers[0] || 'text'
        const texts = rows.map(r => (r[col] ?? '')).filter(Boolean)
        resolve({ text: texts.join('\n'), count: texts.length, headers })
      },
      error: (err) => reject(err),
    })
  })
}

