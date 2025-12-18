import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface PredictionResult {
  filename: string;
  prediction_class: string;
  confidence: number;
  heatmap_base64?: string;
  original_image_base64?: string;
}

@Injectable({
  providedIn: 'root'
})
export class ApiService {
  private baseUrl = 'http://localhost:8000';

  constructor(private http: HttpClient) { }

  predict(file: File): Observable<PredictionResult> {
    const formData = new FormData();
    formData.append('file', file);
    return this.http.post<PredictionResult>(`${this.baseUrl}/predict`, formData);
  }

  checkHealth(): Observable<any> {
    return this.http.get(`${this.baseUrl}/health`);
  }
}
