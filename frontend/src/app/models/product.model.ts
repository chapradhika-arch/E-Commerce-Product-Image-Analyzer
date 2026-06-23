export interface RagSource {
  category: string;
  snippet: string;
  score: number;
}

/** Shape returned by the backend for a single analyzed image. */
export interface ProductAnalysis {
  filename: string;
  caption: string;
  title: string;
  description: string;
  tags: string[];
  category: string;
  rag_sources: RagSource[];
  mock: boolean;
  error: string | null;
}

export interface BulkAnalysisResponse {
  count: number;
  results: ProductAnalysis[];
}

/** UI view-model: a product card the user can edit, backed by its preview. */
export interface ProductCard extends ProductAnalysis {
  /** Stable id for *ngFor tracking. */
  id: string;
  /** object URL for the local image preview. */
  previewUrl: string;
  /** comma-joined tags bound to the editable input. */
  tagsText: string;
}
