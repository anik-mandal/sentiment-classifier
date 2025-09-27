import { getDocument, GlobalWorkerOptions, type PDFDocumentProxy } from "pdfjs-dist";
import pdfWorker from "pdfjs-dist/build/pdf.worker.min.mjs?url";

GlobalWorkerOptions.workerSrc = pdfWorker;

type PageSelection = number[] | undefined;

const toUint8Array = async (file: File): Promise<Uint8Array> => {
  const buffer = await file.arrayBuffer();
  return new Uint8Array(buffer);
};

const sanitizePages = (pages: PageSelection, total: number): number[] => {
  if (!pages || !pages.length) {
    return Array.from({ length: total }, (_, index) => index + 1);
  }

  const unique = new Set<number>();
  pages.forEach((page) => {
    if (!Number.isFinite(page)) return;
    const clamped = Math.min(Math.max(Math.floor(page), 1), total);
    unique.add(clamped);
  });

  return Array.from(unique).sort((a, b) => a - b);
};

const pageRangePattern = /^(\d+)(?:\s*-\s*(\d+))?$/;

export const parsePageRange = (input: string, totalPages: number): number[] => {
  if (!input.trim()) {
    return [];
  }

  const segments = input
    .split(/[;,\s]+/)
    .map((segment) => segment.trim())
    .filter(Boolean);

  const pages = new Set<number>();

  for (const segment of segments) {
    const match = segment.match(pageRangePattern);
    if (!match) {
      throw new Error("Use page numbers like '1-3' or '2,4'.");
    }

    const start = Number(match[1]);
    const end = match[2] ? Number(match[2]) : start;

    if (start > end) {
      throw new Error("Page ranges must ascend.");
    }

    for (let page = start; page <= end; page += 1) {
      if (page < 1 || page > totalPages) {
        throw new Error(`Page ${page} is outside of the document (1-${totalPages}).`);
      }
      pages.add(page);
    }
  }

  return Array.from(pages).sort((a, b) => a - b);
};

const collectPageText = async (doc: PDFDocumentProxy, pages: number[]): Promise<string> => {
  const chunks: string[] = [];

  for (const pageNumber of pages) {
    const page = await doc.getPage(pageNumber);
    const content = await page.getTextContent();
    const text = content.items
      .map((item) => ("str" in item ? item.str : ""))
      .join(" ");
    chunks.push(text);
  }

  const combined = chunks.join("\n").replace(/\s+/g, " ").trim();
  return combined;
};

export const extractPdfText = async (
  file: File,
  pages?: PageSelection,
): Promise<{ text: string; totalPages: number }> => {
  const data = await toUint8Array(file);
  const loadingTask = getDocument({ data });
  const doc = await loadingTask.promise;

  try {
    const totalPages = doc.numPages;
    const targetPages = sanitizePages(pages, totalPages);
    const text = await collectPageText(doc, targetPages);
    return { text, totalPages };
  } finally {
    loadingTask.destroy();
  }
};
