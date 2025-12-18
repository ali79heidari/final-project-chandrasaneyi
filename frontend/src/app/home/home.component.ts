import { Component, ChangeDetectorRef } from '@angular/core';
import { CommonModule } from '@angular/common';
import { ApiService, PredictionResult } from '../services/api.service';
import { trigger, transition, style, animate } from '@angular/animations';

@Component({
  selector: 'app-home',
  standalone: true,
  imports: [CommonModule],
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.css'],
  animations: [
    trigger('fadeIn', [
      transition(':enter', [
        style({ opacity: 0, transform: 'translateY(20px)' }),
        animate('0.5s ease-out', style({ opacity: 1, transform: 'translateY(0)' }))
      ])
    ])
  ]
})
export class HomeComponent {
  selectedFile: File | null = null;
  prediction: PredictionResult | null = null;
  loading = false;
  error: string | null = null;
  imagePreview: string | null = null;

  constructor(private apiService: ApiService, private cdr: ChangeDetectorRef) {}

  onFileSelected(event: any) {
    const file = event.target.files[0];
    if (file) {
      this.selectedFile = file;
      this.prediction = null;
      this.error = null;
      
      // Create preview
      const reader = new FileReader();
      reader.onload = () => {
        this.imagePreview = reader.result as string;
        this.cdr.detectChanges();
      };
      reader.readAsDataURL(file);
    }
  }

  async uploadAndPredict() {
    if (!this.selectedFile) return;

    this.loading = true;
    this.error = null;
    this.prediction = null;
    this.cdr.detectChanges();

    this.apiService.predict(this.selectedFile).subscribe({
      next: (result) => {
        console.log('Prediction Result:', result);
        this.prediction = result;
        this.loading = false;
        this.cdr.detectChanges();
      },
      error: (err) => {
        console.error(err);
        this.error = 'خطا در پردازش تصویر. لطفا دوباره تلاش کنید.';
        this.loading = false;
        this.cdr.detectChanges();
      }
    });
  }

  reset() {
    this.selectedFile = null;
    this.prediction = null;
    this.imagePreview = null;
  }
}
