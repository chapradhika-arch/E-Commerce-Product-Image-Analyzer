import { Component, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { AnalyzerService, HealthResponse } from './services/analyzer.service';
import { ProductCard } from './models/product.model';
import { cardsToCsv, downloadCsv } from './utils/csv';

let cardSeq = 0;

@Component({
  selector: 'app-root',
  imports: [FormsModule],
  templateUrl: './app.html',
  styleUrl: './app.css',
})
export class App {
  private analyzer = inject(AnalyzerService);

  readonly cards = signal<ProductCard[]>([]);
  readonly pending = signal<{ name: string; previewUrl: string }[]>([]);
  readonly loading = signal(false);
  readonly errorMsg = signal<string | null>(null);
  readonly health = signal<HealthResponse | null>(null);
  readonly dragOver = signal(false);

  readonly hasResults = computed(() => this.cards().length > 0);
  readonly hasPending = computed(() => this.pending().length > 0);

  constructor() {
    this.analyzer.health().subscribe({
      next: (h) => this.health.set(h),
      error: () => this.health.set(null),
    });
  }

  // ---- file selection ------------------------------------------------ //
  onFilesSelected(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files) {
      this.addFiles(Array.from(input.files));
    }
    input.value = '';
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    const files = event.dataTransfer?.files;
    if (files) {
      this.addFiles(Array.from(files).filter((f) => f.type.startsWith('image/')));
    }
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(true);
  }

  onDragLeave(): void {
    this.dragOver.set(false);
  }

  private selectedFiles: File[] = [];

  /** Max images per analyze request (matches the backend limit). */
  readonly maxUploads = 20;

  private addFiles(files: File[]): void {
    if (!files.length) return;
    const room = this.maxUploads - this.selectedFiles.length;
    if (room <= 0) {
      this.errorMsg.set(`You can analyze up to ${this.maxUploads} images at a time.`);
      return;
    }
    const accepted = files.slice(0, room);
    if (accepted.length < files.length) {
      this.errorMsg.set(
        `Only the first ${this.maxUploads} images were added (limit per batch).`,
      );
    } else {
      this.errorMsg.set(null);
    }
    this.selectedFiles.push(...accepted);
    this.pending.update((p) => [
      ...p,
      ...accepted.map((f) => ({ name: f.name, previewUrl: URL.createObjectURL(f) })),
    ]);
  }

  clearPending(): void {
    this.pending().forEach((p) => URL.revokeObjectURL(p.previewUrl));
    this.pending.set([]);
    this.selectedFiles = [];
  }

  // ---- analysis ------------------------------------------------------ //
  analyze(): void {
    if (!this.selectedFiles.length || this.loading()) return;
    this.loading.set(true);
    this.errorMsg.set(null);

    const files = [...this.selectedFiles];
    const previews = new Map(this.pending().map((p) => [p.name, p.previewUrl]));

    this.analyzer.analyzeBulk(files).subscribe({
      next: (res) => {
        const newCards: ProductCard[] = res.results.map((r) => ({
          ...r,
          id: `card-${cardSeq++}`,
          previewUrl: previews.get(r.filename) ?? '',
          tagsText: r.tags.join(', '),
        }));
        this.cards.update((c) => [...c, ...newCards]);
        // pending previews are now owned by the cards; don't revoke them.
        this.pending.set([]);
        this.selectedFiles = [];
        this.loading.set(false);
      },
      error: (err) => {
        this.errorMsg.set(
          'Analysis failed. Is the backend running on ' +
            'http://localhost:8000? ' +
            (err?.message ?? ''),
        );
        this.loading.set(false);
      },
    });
  }

  // ---- card editing -------------------------------------------------- //
  removeCard(id: string): void {
    const card = this.cards().find((c) => c.id === id);
    if (card?.previewUrl) URL.revokeObjectURL(card.previewUrl);
    this.cards.update((c) => c.filter((x) => x.id !== id));
  }

  clearAll(): void {
    this.cards().forEach((c) => c.previewUrl && URL.revokeObjectURL(c.previewUrl));
    this.cards.set([]);
  }

  updateField(id: string, field: 'title' | 'description' | 'category', value: string): void {
    this.cards.update((cards) =>
      cards.map((c) => (c.id === id ? { ...c, [field]: value } : c)),
    );
  }

  updateTags(id: string, value: string): void {
    this.cards.update((cards) =>
      cards.map((c) =>
        c.id === id
          ? {
              ...c,
              tagsText: value,
              tags: value
                .split(',')
                .map((t) => t.trim())
                .filter(Boolean),
            }
          : c,
      ),
    );
  }

  // ---- export -------------------------------------------------------- //
  exportCsv(): void {
    if (!this.cards().length) return;
    const stamp = new Date().toISOString().slice(0, 10);
    downloadCsv(cardsToCsv(this.cards()), `products-${stamp}.csv`);
  }
}
