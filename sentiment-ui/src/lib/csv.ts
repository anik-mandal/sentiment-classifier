import Papa from "papaparse";

export type CsvParseResult = {
  rows: string[];
  preview: string[];
};

const rowToText = (row: unknown): string => {
  if (Array.isArray(row)) {
    return row
      .map((cell) => (cell == null ? "" : String(cell).trim()))
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  if (typeof row === "object" && row !== null) {
    return Object.values(row)
      .map((cell) => (cell == null ? "" : String(cell).trim()))
      .filter(Boolean)
      .join(" ")
      .trim();
  }

  return String(row ?? "").trim();
};

export const extractCsvRows = (file: File): Promise<CsvParseResult> => {
  return new Promise((resolve, reject) => {
    Papa.parse(file, {
      skipEmptyLines: "greedy",
      complete: ({ data, errors }) => {
        if (errors.length) {
          reject(new Error(errors[0].message));
          return;
        }

        const rows = (data as unknown[])
          .map(rowToText)
          .filter((entry) => entry.length > 0);

        if (!rows.length) {
          reject(new Error("No readable text rows were found in the CSV."));
          return;
        }

        resolve({
          rows,
          preview: rows.slice(0, 3),
        });
      },
      error: (error) => {
        reject(error);
      },
    });
  });
};
