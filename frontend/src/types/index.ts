export interface Classroom {
  id: string;
  name: string;
  code: string;
  teacher_id: string;
  teacher_name: string;
  department: string;
  description?: string | null;
  active_students_count?: number;
  assigned_problems_count?: number;
}

export interface Assignment {
  id: string;
  classroom_id?: string | null;
  classroom_name?: string | null;
  title: string;
  description: string;
  starter_code: string;
  difficulty: 'easy' | 'medium' | 'hard';
  concept_tags?: string[] | null;
  created_by: string;
  teacher_id?: string | null;
  sample_input?: string | null;
  sample_output?: string | null;
  created_at: string;
}

export type UserRole = 'student' | 'teacher';

export interface AuthUser {
  role: UserRole;
  id: string;
  name: string;
  department?: string;
  classroomId?: string;
  classroomName?: string;
  classroomCode?: string;
  teacherId?: string;
  teacherName?: string;
}

export interface CodeRunResponse {
  success: boolean;
  stdout: string;
  stderr: string;
  exit_code: number;
  execution_time_ms: number;
  sandbox_used: boolean;
  error_type?: string | null;
}

export interface ASTAnalysisResult {
  is_syntax_valid: boolean;
  syntax_error_details?: {
    line: number;
    column: number;
    message: string;
    text: string;
    diagnostic: string;
  } | null;
  variables: Array<{
    name: string;
    line_number: number;
    is_assignment: boolean;
    assigned_value_type?: string | null;
    target_scope: string;
  }>;
  loops: Array<{
    loop_type: string;
    line_number: number;
    end_line_number?: number | null;
    target_variable?: string | null;
    iterable_expression?: string | null;
    has_break: boolean;
    has_continue: boolean;
    nesting_depth: number;
  }>;
  conditions: Array<{
    line_number: number;
    test_expression: string;
    has_elif: boolean;
    has_else: boolean;
    nesting_depth: number;
  }>;
  functions: Array<{
    name: string;
    line_number: number;
    end_line_number?: number | null;
    arguments: string[];
    has_return: boolean;
    return_line_numbers: number[];
    is_recursive: boolean;
    docstring?: string | null;
  }>;
  total_lines: number;
  code_lines: number;
  comment_lines: number;
  cyclomatic_complexity: number;
  max_nesting_depth: number;
  builtin_calls: string[];
  candidate_concepts: Array<{
    concept_key: string;
    concept_name: string;
    line_number: number;
    focal_code: string;
    description: string;
    suggested_difficulty: string;
  }>;
}

export interface CodingSession {
  id: string;
  student_id: string;
  student_name: string;
  classroom_id?: string | null;
  classroom_name?: string | null;
  teacher_id?: string | null;
  teacher_name?: string | null;
  title: string;
  language: string;
  status: string;
  created_at: string;
  updated_at: string;
  submission_count?: number;
  question_count?: number;
  has_report?: boolean;
  latest_verdict?: string | null;
}

export interface CodeSubmission {
  id: string;
  session_id: string;
  code: string;
  ast_summary?: Record<string, any> | null;
  execution_output?: string | null;
  error_output?: string | null;
  is_valid: boolean;
  created_at: string;
}

export interface ProcessEventItem {
  event_type: 'edit' | 'paste' | 'run' | 'submit' | 'idle';
  cursor_line?: number;
  cursor_column?: number;
  delta_chars: number;
  paste_length: number;
  timestamp?: string;
  event_metadata?: Record<string, any>;
}

export interface TimelineMilestone {
  offset_seconds: number;
  event_type: string;
  label: string;
  detail?: string | null;
}

export interface ProcessFeatures {
  session_id: string;
  total_events: number;
  edit_count: number;
  paste_count: number;
  run_count: number;
  error_count?: number;
  total_chars_pasted: number;
  total_chars_edited?: number;
  paste_ratio: number;
  coding_duration_seconds: number;
  average_edit_interval_seconds: number;
  burst_count: number;
  burstiness_ratio?: number;
  time_to_first_run_seconds?: number | null;
  error_recovery_rate?: number;
  typing_speed_cpm?: number;
  process_profile?: string;
  incremental_score?: number;
  debugging_score?: number;
  revision_score?: number;
  testing_score?: number;
  process_score?: number;
  has_process_evidence?: boolean;
  ethical_disclaimer?: string;
  disclosure_notice?: string;
  timeline_events?: TimelineMilestone[];
}

export interface VivaQuestion {
  id: string;
  session_id: string;
  submission_id?: string | null;
  question_text: string;
  focal_code?: string | null;
  concept: string;
  difficulty: string;
  expected_reasoning?: string | null;
  rubric?: Record<string, any> | null;
  misconception_indicators?: string[] | null;
  created_at: string;
  has_answer: boolean;
  assessment?: AnswerAssessment | null;
}

export interface AnswerAssessment {
  id: string;
  question_id: string;
  session_id: string;
  student_answer: string;
  answer_label: 'Correct' | 'Partially Correct' | 'Incorrect' | 'Insufficient Evidence';
  score?: number;
  feedback?: string;
  evaluation_label?: string;
  expected_reasoning?: string | null;
  concepts_correct?: string[] | null;
  concepts_missed?: string[] | null;
  contradictions?: string[] | null;
  supporting_evidence?: string[] | null;
  evaluation_explanation: string;
  uncertainty_score: number;
  needs_instructor_review: boolean;
  instructor_status: string;
  created_at: string;
}

export interface ComprehensionReport {
  id: string;
  session_id: string;
  overall_label: 'Understood' | 'Partial Understanding' | 'Possible Misconception' | 'Pending / Insufficient Evidence';
  confidence_score: number;
  evidence_summary: string;
  evidence_breakdown?: Record<string, any> | null;
  multimodal_scores?: Record<string, number> | null;
  needs_instructor_review?: boolean;
  recommended_interventions?: string[] | null;
  detected_misconceptions?: string[] | null;
  process_features?: Record<string, any> | null;
  rubric_summary?: Record<string, any> | null;
  created_at: string;
}

export interface InstructorOverview {
  total_sessions: number;
  total_submissions: number;
  total_questions: number;
  total_evaluations: number;
  flagged_for_review: number;
  comprehension_breakdown: {
    Understood: number;
    'Partial Understanding': number;
    'Possible Misconception': number;
  };
}

export interface InstructorReview {
  id: string;
  session_id: string;
  reviewer_name: string;
  review_type: string;
  target_id?: string | null;
  assigned_label: string;
  agree_with_ai?: boolean | null;
  feedback_notes?: string | null;
  created_at: string;
}

export interface AblationConfigResult {
  configuration: string;
  config_key: string;
  random_forest: {
    accuracy: number;
    macro_f1: number;
  };
  logistic_regression: {
    accuracy: number;
    macro_f1: number;
  };
  cross_attention_transformer: {
    accuracy: number;
    macro_f1: number;
    confusion_matrix?: number[][];
  };
}

export interface AblationStudyResponse {
  n_samples: number;
  test_samples: number;
  modalities_evaluated: number;
  results: AblationConfigResult[];
}

export interface MLPredictionResponse {
  session_id: string;
  predicted_label: 'Understood' | 'Partial Understanding' | 'Possible Misconception';
  confidence: number;
  probabilities: {
    Understood: number;
    'Partial Understanding': number;
    'Possible Misconception': number;
  };
  modality_contributions: {
    code_norm: number;
    explanation_norm: number;
    process_norm: number;
  };
}

export interface PerClassMetric {
  precision: number;
  recall: number;
  f1: number;
  support: number;
}

export interface InterRaterMetricsResponse {
  cohen_kappa: number;
  kappa_interpretation: string;
  percentage_agreement: number;
  total_comparisons: number;
  accuracy: number;
  macro_f1: number;
  per_class_metrics: Record<string, PerClassMetric>;
  confusion_matrix: {
    labels: string[];
    matrix: number[][];
  };
  flagged_uncertain_rate: number;
  source_breakdown: {
    database_reviews: number;
    benchmark_samples: number;
  };
  classes: string[];
}

export interface MLModelDetail {
  name: string;
  is_available: boolean;
  status: string;
  device?: string | null;
  dimension?: number;
  fallback_feature?: string | null;
  framework?: string;
  is_trained?: boolean;
}

export interface MLStatusResponse {
  status: string;
  pytorch_available: boolean;
  compute_device: string;
  models: {
    codebert: MLModelDetail;
    sentence_transformer: MLModelDetail;
    multimodal_transformer: MLModelDetail & {
      num_classes: number;
      class_labels: string[];
    };
  };
}
