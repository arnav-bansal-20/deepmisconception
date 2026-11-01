import {
  CodingSession,
  CodeSubmission,
  ProcessEventItem,
  ProcessFeatures,
  VivaQuestion,
  AnswerAssessment,
  ComprehensionReport,
  InstructorOverview,
  InstructorReview,
  Classroom,
  Assignment,
  AblationStudyResponse,
  MLPredictionResponse,
  InterRaterMetricsResponse,
} from '../types';

const API_BASE = '/api';

async function request<T>(endpoint: string, options: RequestInit = {}): Promise<T> {
  const url = `${API_BASE}${endpoint}`;
  const response = await fetch(url, {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  });

  if (!response.ok) {
    let errorDetail = `Request failed: ${response.statusText}`;
    try {
      const errJson = await response.json();
      if (errJson.detail) errorDetail = errJson.detail;
    } catch {
      // fallback to status text
    }
    throw new Error(errorDetail);
  }

  if (response.status === 204) {
    return {} as T;
  }

  return response.json();
}

export const api = {
  // Health
  getHealth: () => request<{ status: string; app: string; version: string; sandbox_enabled: boolean }>('/health'),

  // Classrooms & Cohorts
  listClassrooms: (teacherId?: string) => {
    const query = teacherId ? `?teacher_id=${encodeURIComponent(teacherId)}` : '';
    return request<Classroom[]>(`/classrooms${query}`);
  },

  createClassroom: (data: {
    name: string;
    code: string;
    teacher_id: string;
    teacher_name: string;
    department?: string;
    description?: string;
  }) =>
    request<Classroom>('/classrooms', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getClassroomByCode: (code: string) =>
    request<Classroom>(`/classrooms/by-code/${encodeURIComponent(code)}`),

  getClassroom: (id: string) =>
    request<Classroom>(`/classrooms/${encodeURIComponent(id)}`),

  // Sessions
  createSession: (data: {
    student_id: string;
    student_name: string;
    classroom_id?: string;
    classroom_name?: string;
    teacher_id?: string;
    teacher_name?: string;
    title: string;
    language?: string;
  }) =>
    request<CodingSession>('/sessions', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  listSessions: (params?: { classroom_id?: string; teacher_id?: string; student_id?: string }) => {
    const q = new URLSearchParams();
    if (params?.classroom_id) q.set('classroom_id', params.classroom_id);
    if (params?.teacher_id) q.set('teacher_id', params.teacher_id);
    if (params?.student_id) q.set('student_id', params.student_id);
    const qs = q.toString() ? `?${q.toString()}` : '';
    return request<CodingSession[]>(`/sessions${qs}`);
  },

  getSession: (id: string) => request<CodingSession>(`/sessions/${id}`),

  updateSession: (id: string, data: { title?: string; status?: string }) =>
    request<CodingSession>(`/sessions/${id}`, {
      method: 'PATCH',
      body: JSON.stringify(data),
    }),

  deleteSession: (id: string) =>
    request<void>(`/sessions/${id}`, {
      method: 'DELETE',
    }),

  // Submissions
  submitCode: (sessionId: string, code: string) =>
    request<CodeSubmission>('/submissions', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, code }),
    }),

  getSessionSubmissions: (sessionId: string) =>
    request<CodeSubmission[]>(`/submissions/session/${sessionId}`),

  // Telemetry
  sendTelemetryBatch: (sessionId: string, events: ProcessEventItem[]) =>
    request<{ status: string; recorded_events: number }>('/telemetry/batch', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId, events }),
    }),

  getProcessFeatures: (sessionId: string) =>
    request<ProcessFeatures>(`/telemetry/features/${sessionId}`),

  // Code Execution & AST Analysis
  analyzeCode: (code: string) =>
    request<import('../types').ASTAnalysisResult>('/code/analyze', {
      method: 'POST',
      body: JSON.stringify({ code }),
    }),

  runCode: (code: string, timeoutSeconds?: number) =>
    request<import('../types').CodeRunResponse>('/code/run', {
      method: 'POST',
      body: JSON.stringify({ code, timeout_seconds: timeoutSeconds }),
    }),

  // Viva Questions & Assessments
  getVivaQuestions: (sessionId: string) =>
    request<VivaQuestion[]>(`/viva/session/${sessionId}`),

  generateVivaQuestions: (sessionId: string, submissionId?: string, count: number = 3) =>
    request<VivaQuestion[]>('/viva/generate', {
      method: 'POST',
      body: JSON.stringify({
        session_id: sessionId,
        submission_id: submissionId,
        count,
      }),
    }),

  submitVivaAnswer: (questionId: string, studentAnswer: string) =>
    request<AnswerAssessment>('/viva/evaluate', {
      method: 'POST',
      body: JSON.stringify({
        question_id: questionId,
        student_answer: studentAnswer,
      }),
    }),

  // Reports
  getSessionReport: (sessionId: string) =>
    request<ComprehensionReport>(`/reports/session/${sessionId}`),

  generateReport: (sessionId: string) =>
    request<ComprehensionReport>('/reports/generate', {
      method: 'POST',
      body: JSON.stringify({ session_id: sessionId }),
    }),

  // Instructor
  getInstructorOverview: (params?: { classroom_id?: string; teacher_id?: string }) => {
    const q = new URLSearchParams();
    if (params?.classroom_id) q.set('classroom_id', params.classroom_id);
    if (params?.teacher_id) q.set('teacher_id', params.teacher_id);
    const qs = q.toString() ? `?${q.toString()}` : '';
    return request<InstructorOverview>(`/instructor/overview${qs}`);
  },

  submitInstructorReview: (data: {
    session_id: string;
    reviewer_name: string;
    review_type?: string;
    target_id?: string | null;
    assigned_label: string;
    agree_with_ai?: boolean | null;
    feedback_notes?: string | null;
  }) =>
    request<InstructorReview>('/instructor/reviews', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  getSessionReviews: (sessionId: string) =>
    request<InstructorReview[]>(`/instructor/reviews/session/${sessionId}`),

  // Assignments
  listAssignments: (classroomId?: string) => {
    const query = classroomId ? `?classroom_id=${encodeURIComponent(classroomId)}` : '';
    return request<Assignment[]>(`/assignments${query}`);
  },

  createAssignment: (data: {
    title: string;
    description: string;
    starter_code: string;
    difficulty?: string;
    concept_tags?: string[];
    created_by?: string;
    classroom_id?: string | null;
    classroom_name?: string | null;
  }) =>
    request<Assignment>('/assignments', {
      method: 'POST',
      body: JSON.stringify(data),
    }),

  deleteAssignment: (id: string) =>
    request<void>(`/assignments/${id}`, {
      method: 'DELETE',
    }),

  // Machine Learning & Scientific Ablations (Phase 8 & 9)
  getAblationBenchmarks: (samples: number = 180, seed: number = 42) =>
    request<AblationStudyResponse>(`/ml/ablation?samples=${samples}&seed=${seed}`),

  predictSessionML: (sessionId: string) =>
    request<MLPredictionResponse>(`/ml/predict/${encodeURIComponent(sessionId)}`, {
      method: 'POST',
    }),

  getEvaluationMetrics: (params?: {
    classroom_id?: string;
    teacher_id?: string;
    include_benchmark?: boolean;
  }) => {
    const q = new URLSearchParams();
    if (params?.classroom_id) q.set('classroom_id', params.classroom_id);
    if (params?.teacher_id) q.set('teacher_id', params.teacher_id);
    if (params?.include_benchmark !== undefined) {
      q.set('include_benchmark', String(params.include_benchmark));
    }
    const qs = q.toString() ? `?${q.toString()}` : '';
    return request<InterRaterMetricsResponse>(`/instructor/evaluation-metrics${qs}`);
  },
};
