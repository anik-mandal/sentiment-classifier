import * as pdfjsLib from 'pdfjs-dist'
import 'pdfjs-dist/build/pdf.worker.mjs'

export async function extractPdfText(file: File, pages?: number[]): Promise<{ text: string; pageCount: number }>{
  const arrayBuf = await file.arrayBuffer()
  const loadingTask = pdfjsLib.getDocument({ data: arrayBuf })
  const pdf = await loadingTask.promise
  const pageCount = pdf.numPages
  const selected = pages && pages.length ? pages : Array.from({ length: pageCount }, (_, i) => i + 1)
  let fullText = ''
  for (const p of selected) {
    const page = await pdf.getPage(p)
    const content = await page.getTextContent()
    const strings = content.items.map((it: any) => ('str' in it ? it.str : '')).join(' ')
    if (strings.trim()) fullText += strings + '\n'
  }
  return { text: fullText.trim(), pageCount }
}

