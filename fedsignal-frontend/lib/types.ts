export interface Prediction {
  date: string;
  label: string;
  predicted: number;
  actual: number | null;
  regime: string;
}

export interface SentimentPoint {
  date: string;
  score: number;
  label: string;
}

export interface FeatureImportance {
  feature: string;
  importance: number;
}

export interface Metrics {
  oos_r2: number;
  dir_acc: number;
  mae: number;
  n_obs: number;
  n_test: number;
  feature_importances: FeatureImportance[];
}

export interface ExampleStatement {
  text: string;
  score: number;
  label: string;
}
