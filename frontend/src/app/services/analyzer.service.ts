import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { environment } from '../../environments/environment';
import { BulkAnalysisResponse, ProductAnalysis } from '../models/product.model';

export interface HealthResponse {
  status: string;
  mock_mode: boolean;
  device: string;
  models: Record<string, string>;
  knowledge_base_entries: number;
}

@Injectable({ providedIn: 'root' })
export class AnalyzerService {
  private http = inject(HttpClient);
  private base = environment.apiBaseUrl;

  health(): Observable<HealthResponse> {
    return this.http.get<HealthResponse>(`${this.base}/api/health`);
  }

  analyzeBulk(files: File[]): Observable<BulkAnalysisResponse> {
    const form = new FormData();
    for (const file of files) {
      form.append('files', file, file.name);
    }
    return this.http.post<BulkAnalysisResponse>(
      `${this.base}/api/analyze/bulk`,
      form,
    );
  }

  analyzeSingle(file: File): Observable<ProductAnalysis> {
    const form = new FormData();
    form.append('file', file, file.name);
    return this.http.post<ProductAnalysis>(`${this.base}/api/analyze`, form);
  }
}
