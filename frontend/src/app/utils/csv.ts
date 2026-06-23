import { ProductCard } from '../models/product.model';

function escapeCell(value: string): string {
  const v = value ?? '';
  // Wrap in quotes and double any embedded quotes (RFC 4180).
  if (/[",\n\r]/.test(v)) {
    return `"${v.replace(/"/g, '""')}"`;
  }
  return v;
}

/** Serialize the edited product cards to a CSV string. */
export function cardsToCsv(cards: ProductCard[]): string {
  const headers = [
    'filename',
    'title',
    'category',
    'description',
    'tags',
    'caption',
  ];
  const rows = cards.map((c) =>
    [
      c.filename,
      c.title,
      c.category,
      c.description,
      c.tagsText,
      c.caption,
    ]
      .map((cell) => escapeCell(String(cell)))
      .join(','),
  );
  return [headers.join(','), ...rows].join('\r\n');
}

/** Trigger a browser download of the given CSV text. */
export function downloadCsv(csv: string, filename = 'products.csv'): void {
  // Prepend BOM so Excel detects UTF-8.
  const blob = new Blob(['﻿' + csv], { type: 'text/csv;charset=utf-8;' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
