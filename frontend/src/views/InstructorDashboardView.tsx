import React, { useState, useEffect } from 'react';
import { 
  ShieldCheck, 
  Users, 
  FileCode, 
  HelpCircle, 
  Award, 
  AlertTriangle, 
  CheckCircle2, 
  RefreshCw,
  PlusCircle,
  BookOpen,
  Trash2,
  FolderPlus,
  Hash,
  School,
} from 'lucide-react';
import { 
  CodingSession, 
  InstructorOverview, 
  Assignment, 
  VivaQuestion, 
  CodeSubmission, 
  ComprehensionReport,
  Classroom,
  AuthUser,
  ProcessFeatures,
} from '../types';
import { api } from '../services/api';
import { ProcessTimeline } from '../components/ProcessTimeline';

interface InstructorDashboardViewProps {
  onSelectSession: (session: CodingSession) => void;
  currentUser?: AuthUser | null;
}

export const InstructorDashboardView: React.FC<InstructorDashboardViewProps> = ({
  onSelectSession,
  currentUser,
}) => {
  const [activeTab, setActiveTab] = useState<'reports' | 'authoring'>('reports');
  const [overview, setOverview] = useState<InstructorOverview | null>(null);
  const [sessions, setSessions] = useState<CodingSession[]>([]);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [classrooms, setClassrooms] = useState<Classroom[]>([]);
  const [selectedClassroomId, setSelectedClassroomId] = useState<string>('all');
  const [isLoading, setIsLoading] = useState(false);

  // Detailed Student Report Modal State
  const [inspectingSession, setInspectingSession] = useState<CodingSession | null>(null);
  const [studentReport, setStudentReport] = useState<ComprehensionReport | null>(null);
  const [studentQuestions, setStudentQuestions] = useState<VivaQuestion[]>([]);
  const [studentSubmissions, setStudentSubmissions] = useState<CodeSubmission[]>([]);
  const [studentFeatures, setStudentFeatures] = useState<ProcessFeatures | null>(null);
  const [isLoadingReport, setIsLoadingReport] = useState(false);

  // Annotation Form State
  const reviewerName = currentUser?.name || 'Instructor';
  const [assignedLabel, setAssignedLabel] = useState('Understood');
  const [agreeWithAi, setAgreeWithAi] = useState(true);
  const [feedbackNotes, setFeedbackNotes] = useState('');
  const [isSubmittingReview, setIsSubmittingReview] = useState(false);
  const [reviewSuccess, setReviewSuccess] = useState(false);

  // New Assignment Creation Form State
  const [newTitle, setNewTitle] = useState('');
  const [newDescription, setNewDescription] = useState('');
  const [newStarterCode, setNewStarterCode] = useState('def solution():\n    pass\n');
  const [newDifficulty, setNewDifficulty] = useState<'easy' | 'medium' | 'hard'>('medium');
  const [newTags, setNewTags] = useState('loops, conditions, recursion');
  const [newSampleInput, setNewSampleInput] = useState('');
  const [newSampleOutput, setNewSampleOutput] = useState('');
  const [targetClassroomId, setTargetClassroomId] = useState<string>('');
  const [isCreatingAssignment, setIsCreatingAssignment] = useState(false);
  const [assignmentSuccess, setAssignmentSuccess] = useState(false);

  // New Classroom Modal / Card State
  const [showCreateClassModal, setShowCreateClassModal] = useState(false);
  const [newClassName, setNewClassName] = useState('');
  const [newClassCode, setNewClassCode] = useState('');
  const [newClassDesc, setNewClassDesc] = useState('');
  const [classCreateError, setClassCreateError] = useState<string | null>(null);
  const [isCreatingClass, setIsCreatingClass] = useState(false);

  const fetchData = async () => {
    setIsLoading(true);
    try {
      const classFilter = selectedClassroomId === 'all' ? undefined : selectedClassroomId;
      const [ovData, sessData, assignData, classList] = await Promise.allSettled([
        api.getInstructorOverview({ classroom_id: classFilter, teacher_id: currentUser?.id }),
        api.listSessions({ classroom_id: classFilter, teacher_id: currentUser?.id }),
        api.listAssignments({
          classroomId: classFilter,
          teacherId: currentUser?.id,
          createdBy: currentUser?.name,
        }),
        api.listClassrooms(currentUser?.id),
      ]);

      if (ovData.status === 'fulfilled') setOverview(ovData.value);
      if (sessData.status === 'fulfilled') setSessions(sessData.value);
      if (assignData.status === 'fulfilled') setAssignments(assignData.value);
      if (classList.status === 'fulfilled') {
        setClassrooms(classList.value);
        if (classList.value.length > 0 && !targetClassroomId) {
          setTargetClassroomId(classList.value[0].id);
        }
      }
    } catch (err) {
      console.error('Failed to load instructor metrics', err);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchData();
  }, [selectedClassroomId]);

  const handleInspectStudentReport = async (session: CodingSession) => {
    setInspectingSession(session);
    setIsLoadingReport(true);
    setReviewSuccess(false);
    setAssignedLabel(session.latest_verdict || 'Understood');
    setFeedbackNotes('');

    try {
      const [repData, qData, subData, featData] = await Promise.allSettled([
        api.getSessionReport(session.id),
        api.getVivaQuestions(session.id),
        api.getSessionSubmissions(session.id),
        api.getProcessFeatures(session.id),
      ]);

      if (repData.status === 'fulfilled') setStudentReport(repData.value);
      if (qData.status === 'fulfilled') setStudentQuestions(qData.value);
      if (subData.status === 'fulfilled') setStudentSubmissions(subData.value);
      if (featData.status === 'fulfilled') setStudentFeatures(featData.value);
    } catch (err) {
      console.error('Failed to load student report details', err);
    } finally {
      setIsLoadingReport(false);
    }
  };

  const handleSubmitInstructorReview = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!inspectingSession) return;

    setIsSubmittingReview(true);
    try {
      await api.submitInstructorReview({
        session_id: inspectingSession.id,
        reviewer_name: reviewerName,
        review_type: 'comprehensive_audit',
        assigned_label: assignedLabel,
        agree_with_ai: agreeWithAi,
        feedback_notes: feedbackNotes.trim() || 'Reviewed by instructor',
      });
      setReviewSuccess(true);
      setTimeout(() => {
        setInspectingSession(null);
        fetchData();
      }, 1500);
    } catch (err) {
      console.error('Failed to save instructor review', err);
    } finally {
      setIsSubmittingReview(false);
    }
  };

  const handleCreateAssignment = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newTitle.trim() || !newDescription.trim() || !newStarterCode.trim()) return;

    setIsCreatingAssignment(true);
    try {
      const tags = newTags.split(',').map((t) => t.trim()).filter(Boolean);
      const chosenClass = classrooms.find((c) => c.id === targetClassroomId);

      await api.createAssignment({
        title: newTitle.trim(),
        description: newDescription.trim(),
        starter_code: newStarterCode,
        difficulty: newDifficulty,
        concept_tags: tags,
        created_by: reviewerName,
        teacher_id: currentUser?.id,
        classroom_id: chosenClass ? chosenClass.id : null,
        classroom_name: chosenClass ? chosenClass.name : null,
        sample_input: newSampleInput.trim() || undefined,
        sample_output: newSampleOutput.trim() || undefined,
      });

      setAssignmentSuccess(true);
      setNewTitle('');
      setNewDescription('');
      setNewStarterCode('def solution():\n    pass\n');
      setNewSampleInput('');
      setNewSampleOutput('');
      const classFilter = selectedClassroomId === 'all' ? undefined : selectedClassroomId;
      const updated = await api.listAssignments({
        classroomId: classFilter,
        teacherId: currentUser?.id,
        createdBy: currentUser?.name,
      });
      setAssignments(updated);
      setTimeout(() => setAssignmentSuccess(false), 2000);
    } catch (err) {
      console.error('Failed to create assignment', err);
    } finally {
      setIsCreatingAssignment(false);
    }
  };

  const handleCreateNewClassroom = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!newClassName.trim() || !newClassCode.trim()) return;

    setIsCreatingClass(true);
    setClassCreateError(null);
    try {
      const created = await api.createClassroom({
        name: newClassName.trim(),
        code: newClassCode.trim(),
        teacher_id: currentUser?.id || 'FAC-CS-804',
        teacher_name: reviewerName,
        department: currentUser?.department || 'Computer Science & Engineering',
        description: newClassDesc.trim() || undefined,
      });
      setClassrooms((prev) => [...prev, created]);
      setTargetClassroomId(created.id);
      setShowCreateClassModal(false);
      setNewClassName('');
      setNewClassCode('');
      setNewClassDesc('');
    } catch (err: any) {
      setClassCreateError(err.message || 'Failed to create classroom group');
    } finally {
      setIsCreatingClass(false);
    }
  };

  const handleDeleteAssignment = async (id: string) => {
    if (!window.confirm('Delete this coding assignment?')) return;
    try {
      await api.deleteAssignment(id);
      setAssignments((prev) => prev.filter((a) => a.id !== id));
    } catch (err) {
      console.error('Failed to delete assignment', err);
    }
  };

  return (
    <div className="max-w-7xl mx-auto space-y-8 px-4 sm:px-6 py-6">
      {/* Top Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-slate-200 p-6 rounded-2xl shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-700 border border-indigo-200 font-bold uppercase">
              Instructor Studio & Portal
            </span>
            <span className="text-xs text-slate-500 font-mono">Supervisor: {reviewerName}</span>
          </div>
          <h2 className="text-2xl font-black text-slate-900 mt-1">Academic Supervision Dashboard</h2>
          <p className="text-xs text-slate-500 mt-0.5">
            Supervise designated student cohorts, publish targeted coding questions, and review comprehension reports.
          </p>
        </div>

        {/* View Switcher: Reports vs Authoring */}
        <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200 text-xs">
          <button
            onClick={() => setActiveTab('reports')}
            className={`px-3.5 py-2 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
              activeTab === 'reports'
                ? 'bg-white text-indigo-800 shadow-sm border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <BookOpen className="w-4 h-4" />
            <span>Student Reports ({sessions.length})</span>
          </button>

          <button
            onClick={() => setActiveTab('authoring')}
            className={`px-3.5 py-2 rounded-lg font-bold transition-all flex items-center gap-1.5 ${
              activeTab === 'authoring'
                ? 'bg-white text-indigo-800 shadow-sm border border-slate-200'
                : 'text-slate-600 hover:text-slate-900'
            }`}
          >
            <PlusCircle className="w-4 h-4" />
            <span>My Published Questions ({assignments.length})</span>
          </button>

          <button
            onClick={fetchData}
            disabled={isLoading}
            className="p-2 text-slate-500 hover:text-slate-800 transition-colors"
            title="Refresh"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          </button>
        </div>
      </div>

      {activeTab === 'reports' && (
        <div className="space-y-6">
          {/* Cohort Selector / Filter Bar */}
          <div className="bg-white border border-slate-200 p-4 rounded-2xl shadow-sm flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className="text-xs font-mono font-bold uppercase text-slate-500 flex items-center gap-1.5">
                <Users className="w-4 h-4 text-indigo-600" />
                <span>Filter by Student Cohort:</span>
              </span>
              <select
                value={selectedClassroomId}
                onChange={(e) => setSelectedClassroomId(e.target.value)}
                className="bg-slate-50 border border-slate-300 rounded-xl px-3 py-1.5 text-xs font-bold text-slate-800 focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                <option value="all">All Guided Cohorts ({classrooms.length} groups)</option>
                {classrooms.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.code} — {c.name}
                  </option>
                ))}
              </select>
            </div>

            <div className="text-xs text-slate-500 font-mono">
              Showing reports for: <strong className="text-slate-800">{selectedClassroomId === 'all' ? 'All Guided Cohorts' : classrooms.find((c) => c.id === selectedClassroomId)?.name}</strong>
            </div>
          </div>

          {/* Aggregate Metrics Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Students</span>
                <Users className="w-4 h-4 text-sky-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">{overview?.total_sessions || 0}</div>
              <div className="text-[10px] text-slate-500 mt-1">Active candidate sessions</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Submissions</span>
                <FileCode className="w-4 h-4 text-emerald-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">{overview?.total_submissions || 0}</div>
              <div className="text-[10px] text-slate-500 mt-1">AST snapshots executed</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">AI Viva Qs</span>
                <HelpCircle className="w-4 h-4 text-indigo-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">{overview?.total_questions || 0}</div>
              <div className="text-[10px] text-slate-500 mt-1">Generated viva items</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Evaluations</span>
                <Award className="w-4 h-4 text-amber-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">{overview?.total_evaluations || 0}</div>
              <div className="text-[10px] text-slate-500 mt-1">Student answers scored</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Flagged</span>
                <AlertTriangle className="w-4 h-4 text-rose-600" />
              </div>
              <div className="text-2xl font-bold text-rose-600">{overview?.flagged_for_review || 0}</div>
              <div className="text-[10px] text-slate-500 mt-1">High uncertainty / conflict</div>
            </div>
          </div>

          {/* Student Cohorts Table */}
          <div className="bg-white border border-slate-200 rounded-2xl overflow-hidden shadow-sm">
            <div className="p-4 bg-slate-50 border-b border-slate-200 flex items-center justify-between">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <Users className="w-4 h-4 text-indigo-600" />
                Student Comprehension Records & Reports
              </h3>
              <span className="text-xs text-slate-500 font-mono">
                {sessions.length} student sessions recorded
              </span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left text-xs">
                <thead className="bg-slate-50 text-slate-500 font-mono uppercase text-[10px] border-b border-slate-200">
                  <tr>
                    <th className="p-4">Student</th>
                    <th className="p-4">Cohort Group</th>
                    <th className="p-4">Problem Task</th>
                    <th className="p-4">Submissions</th>
                    <th className="p-4">Viva Qs</th>
                    <th className="p-4">Diagnosis Verdict</th>
                    <th className="p-4">Status</th>
                    <th className="p-4 text-right">Actions</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 text-slate-700">
                  {sessions.length === 0 ? (
                    <tr>
                      <td colSpan={8} className="p-8 text-center text-slate-400 italic">
                        No student sessions recorded yet for this cohort. Students can join using the class code and submit code.
                      </td>
                    </tr>
                  ) : (
                    sessions.map((s) => (
                      <tr key={s.id} className="hover:bg-slate-50/80 transition-colors">
                        <td className="p-4 font-medium text-slate-900">
                          <div className="font-bold">{s.student_name}</div>
                          <span className="text-[11px] text-slate-500 font-mono">{s.student_id}</span>
                        </td>
                        <td className="p-4 font-mono text-slate-700">
                          <span className="px-2 py-0.5 rounded bg-sky-50 text-sky-800 border border-sky-200 text-[10px] font-bold">
                            {s.classroom_name || 'Group Alpha'}
                          </span>
                        </td>
                        <td className="p-4 text-slate-800 font-medium">{s.title}</td>
                        <td className="p-4 font-mono">{s.submission_count || 0}</td>
                        <td className="p-4 font-mono">{s.question_count || 0}</td>
                        <td className="p-4">
                          {s.latest_verdict ? (
                            <span className={`inline-flex px-2 py-0.5 rounded-full text-[10px] font-bold ${
                              s.latest_verdict === 'Understood'
                                ? 'bg-emerald-50 text-emerald-800 border border-emerald-300'
                                : s.latest_verdict === 'Partial Understanding'
                                ? 'bg-amber-50 text-amber-800 border border-amber-300'
                                : s.latest_verdict === 'Pending / Insufficient Evidence'
                                ? 'bg-slate-100 text-slate-800 border border-slate-300'
                                : 'bg-rose-50 text-rose-800 border border-rose-300'
                            }`}>
                              {s.latest_verdict}
                            </span>
                          ) : (
                            <span className="text-slate-400 italic">Pending Viva</span>
                          )}
                        </td>
                        <td className="p-4 font-mono capitalize">
                          <span className={`px-2 py-0.5 rounded text-[10px] font-semibold ${
                            s.status === 'reviewed' ? 'bg-indigo-50 text-indigo-800 border border-indigo-200' : 'text-slate-500'
                          }`}>
                            {s.status}
                          </span>
                        </td>
                        <td className="p-4 text-right">
                          <button
                            onClick={() => handleInspectStudentReport(s)}
                            className="px-3 py-1.5 bg-indigo-50 hover:bg-indigo-600 text-indigo-700 hover:text-white border border-indigo-200 hover:border-indigo-600 rounded-lg font-bold text-[11px] transition-all shadow-sm"
                          >
                            Inspect Report
                          </button>
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {/* TAB 2: AUTHOR CODING QUESTIONS */}
      {activeTab === 'authoring' && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Form: Create New Coding Problem */}
          <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-5">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <PlusCircle className="w-5 h-5 text-indigo-600" />
                <h3 className="font-bold text-slate-900 text-base">Author New Coding Question</h3>
              </div>
              <button
                type="button"
                onClick={() => setShowCreateClassModal(true)}
                className="text-[11px] text-indigo-600 hover:text-indigo-800 font-bold flex items-center gap-1"
              >
                <FolderPlus className="w-3.5 h-3.5" />
                <span>+ Create Group</span>
              </button>
            </div>

            {assignmentSuccess && (
              <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-800 rounded-xl text-xs flex items-center gap-2 animate-fadeIn font-semibold">
                <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                <span>Coding question successfully published to designated student cohort!</span>
              </div>
            )}

            <form onSubmit={handleCreateAssignment} className="space-y-4 text-xs">
              {/* Target Classroom Selection */}
              <div>
                <label className="block font-mono font-bold uppercase tracking-wider text-slate-700 mb-1">
                  Assign to Student Cohort / Group:
                </label>
                <select
                  value={targetClassroomId}
                  onChange={(e) => setTargetClassroomId(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
                >
                  {classrooms.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.name} ({c.code}) — {c.teacher_name}
                    </option>
                  ))}
                  <option value="">All Cohorts (Open Global Access)</option>
                </select>
                <p className="text-[10px] text-slate-500 mt-1">
                  Only students enrolled in this group will see this problem in their task selector.
                </p>
              </div>

              <div>
                <label className="block font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                  Problem Title
                </label>
                <input
                  type="text"
                  required
                  value={newTitle}
                  onChange={(e) => setNewTitle(e.target.value)}
                  placeholder="e.g. Inverted Binary Tree Traversal"
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl px-3.5 py-2.5 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium"
                />
              </div>

              <div>
                <label className="block font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                  Problem Description & Requirements
                </label>
                <textarea
                  required
                  rows={4}
                  value={newDescription}
                  onChange={(e) => setNewDescription(e.target.value)}
                  placeholder="Describe the logic requirement, edge cases, and expected algorithmic behavior..."
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl p-3 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-sans leading-relaxed"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Difficulty Level
                  </label>
                  <select
                    value={newDifficulty}
                    onChange={(e) => setNewDifficulty(e.target.value as any)}
                    className="w-full bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-800 font-medium focus:outline-none focus:ring-2 focus:ring-indigo-500"
                  >
                    <option value="easy">Easy (Foundational)</option>
                    <option value="medium">Medium (Control Flow / Recursion)</option>
                    <option value="hard">Hard (Complex Invariants)</option>
                  </select>
                </div>

                <div>
                  <label className="block font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                    Concept Tags (comma separated)
                  </label>
                  <input
                    type="text"
                    value={newTags}
                    onChange={(e) => setNewTags(e.target.value)}
                    placeholder="e.g. loops, modulo, recursion"
                    className="w-full bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono"
                  />
                </div>
              </div>

              {/* Sample Test Case: Input and Output */}
              <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 space-y-3">
                <div className="flex items-center justify-between">
                  <label className="block font-mono font-bold uppercase tracking-wider text-slate-800 text-[11px]">
                    Sample Test Case (Input &amp; Expected Output)
                  </label>
                  <span className="text-[10px] font-mono text-indigo-700 bg-indigo-50 px-2 py-0.5 rounded font-semibold border border-indigo-100">
                    Optional
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block font-mono font-semibold uppercase tracking-wider text-slate-600 text-[10px] mb-1">
                      Sample Input
                    </label>
                    <textarea
                      rows={2}
                      value={newSampleInput}
                      onChange={(e) => setNewSampleInput(e.target.value)}
                      placeholder="e.g. n = 5 or [1, 2, 3]"
                      className="w-full bg-white border border-slate-300 rounded-xl p-2.5 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-[11px]"
                    />
                  </div>
                  <div>
                    <label className="block font-mono font-semibold uppercase tracking-wider text-slate-600 text-[10px] mb-1">
                      Expected Output
                    </label>
                    <textarea
                      rows={2}
                      value={newSampleOutput}
                      onChange={(e) => setNewSampleOutput(e.target.value)}
                      placeholder="e.g. 26 or True"
                      className="w-full bg-white border border-slate-300 rounded-xl p-2.5 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-mono text-[11px]"
                    />
                  </div>
                </div>
                <p className="text-[10px] text-slate-500">
                  Providing a sample input and expected output helps students verify their solution and tests program correctness.
                </p>
              </div>

              <div>
                <label className="block font-mono font-semibold uppercase tracking-wider text-slate-700 mb-1">
                  Starter Python Code Template
                </label>
                <textarea
                  required
                  rows={6}
                  value={newStarterCode}
                  onChange={(e) => setNewStarterCode(e.target.value)}
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl p-3 text-slate-900 font-mono text-[11px] focus:outline-none focus:ring-2 focus:ring-indigo-500 leading-relaxed"
                />
              </div>

              <button
                type="submit"
                disabled={isCreatingAssignment}
                className="w-full py-3 px-4 rounded-xl font-bold bg-indigo-600 hover:bg-indigo-700 text-white shadow-md shadow-indigo-500/20 transition-all flex items-center justify-center gap-2"
              >
                <PlusCircle className="w-4 h-4" />
                <span>{isCreatingAssignment ? 'Publishing...' : 'Publish to Target Cohort'}</span>
              </button>
            </form>
          </div>

          {/* Right Side: Published Assignments List */}
          <div className="space-y-4">
            <h3 className="font-bold text-slate-900 text-base flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-indigo-600" />
              <span>My Published Coding Questions ({assignments.length})</span>
            </h3>

            {assignments.length === 0 ? (
              <div className="bg-white border border-slate-200 rounded-2xl p-8 text-center space-y-2">
                <BookOpen className="w-8 h-8 text-slate-300 mx-auto" />
                <p className="text-xs font-semibold text-slate-700">No coding questions published by you yet</p>
                <p className="text-[11px] text-slate-400 max-w-sm mx-auto">
                  Use the authoring form on the left to create and publish programming questions with sample test inputs and outputs for your student cohorts.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {assignments.map((a) => (
                  <div key={a.id} className="bg-white border border-slate-200 rounded-2xl p-5 shadow-sm space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold ${
                          a.difficulty === 'easy'
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                            : a.difficulty === 'medium'
                            ? 'bg-amber-50 text-amber-800 border border-amber-200'
                            : 'bg-rose-50 text-rose-800 border border-rose-200'
                        }`}>
                          {a.difficulty}
                        </span>
                        <h4 className="font-bold text-slate-900 text-sm">{a.title}</h4>
                        {a.classroom_name && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-50 text-sky-800 border border-sky-200 font-bold">
                            Assigned to: {a.classroom_name}
                          </span>
                        )}
                      </div>

                      <button
                        onClick={() => handleDeleteAssignment(a.id)}
                        className="p-1 text-slate-400 hover:text-rose-600 transition-colors"
                        title="Delete assignment"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>

                    <p className="text-xs text-slate-600 leading-relaxed">{a.description}</p>

                    {/* Sample Test Case Preview */}
                    {(a.sample_input || a.sample_output) && (
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-[11px] font-mono grid grid-cols-1 sm:grid-cols-2 gap-2.5">
                        {a.sample_input && (
                          <div>
                            <span className="text-[10px] uppercase font-bold text-slate-500 block mb-1">Sample Input:</span>
                            <div className="text-slate-800 bg-white px-2.5 py-1.5 rounded-lg border border-slate-200 font-mono text-[11px] whitespace-pre-wrap">
                              {a.sample_input}
                            </div>
                          </div>
                        )}
                        {a.sample_output && (
                          <div>
                            <span className="text-[10px] uppercase font-bold text-emerald-700 block mb-1">Expected Output:</span>
                            <div className="text-emerald-800 bg-white px-2.5 py-1.5 rounded-lg border border-emerald-200 font-mono text-[11px] whitespace-pre-wrap font-semibold">
                              {a.sample_output}
                            </div>
                          </div>
                        )}
                      </div>
                    )}

                    {/* Concept Tags */}
                    {a.concept_tags && a.concept_tags.length > 0 && (
                      <div className="flex flex-wrap gap-1.5 pt-1">
                        {a.concept_tags.map((tag, i) => (
                          <span key={i} className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-100 border border-slate-200 text-indigo-700 font-semibold">
                            #{tag}
                          </span>
                        ))}
                      </div>
                    )}

                    {/* Starter code preview */}
                    <pre className="bg-slate-50 text-[11px] font-mono p-3 rounded-xl border border-slate-200 text-slate-900 overflow-x-auto max-h-24">
                      {a.starter_code}
                    </pre>

                    <div className="text-[10px] text-slate-500 font-mono flex items-center justify-between pt-1">
                      <span>Created by: {a.created_by}</span>
                      <span>{new Date(a.created_at).toLocaleDateString()}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      {/* MODAL: CREATE NEW CLASSROOM GROUP */}
      {showCreateClassModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fadeIn">
          <div className="bg-white border border-slate-200 rounded-3xl max-w-md w-full p-6 shadow-2xl space-y-4">
            <div className="flex items-center justify-between border-b border-slate-100 pb-3">
              <div className="flex items-center gap-2">
                <FolderPlus className="w-5 h-5 text-indigo-600" />
                <h3 className="font-bold text-slate-900 text-base">Create Student Cohort</h3>
              </div>
              <button
                onClick={() => setShowCreateClassModal(false)}
                className="text-slate-400 hover:text-slate-700 font-bold"
              >
                &times;
              </button>
            </div>

            {classCreateError && (
              <p className="text-xs text-rose-600 bg-rose-50 border border-rose-200 p-2 rounded-xl">
                {classCreateError}
              </p>
            )}

            <form onSubmit={handleCreateNewClassroom} className="space-y-3 text-xs">
              <div>
                <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                  Cohort / Class Name
                </label>
                <input
                  type="text"
                  required
                  value={newClassName}
                  onChange={(e) => setNewClassName(e.target.value)}
                  placeholder="e.g. CS101: Python Foundations - Section C"
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-medium"
                />
              </div>

              <div>
                <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                  Class Enrollment Code (Given to Students)
                </label>
                <input
                  type="text"
                  required
                  value={newClassCode}
                  onChange={(e) => setNewClassCode(e.target.value.toUpperCase())}
                  placeholder="e.g. CS101-SEC-C"
                  className="w-full uppercase font-mono bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500 font-bold"
                />
              </div>

              <div>
                <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                  Cohort Description (Optional)
                </label>
                <textarea
                  rows={2}
                  value={newClassDesc}
                  onChange={(e) => setNewClassDesc(e.target.value)}
                  placeholder="Cohort objectives, syllabus focus..."
                  className="w-full bg-slate-50 border border-slate-300 rounded-xl p-2.5 text-slate-900 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                />
              </div>

              <div className="pt-2 flex items-center justify-end gap-2">
                <button
                  type="button"
                  onClick={() => setShowCreateClassModal(false)}
                  className="px-4 py-2 text-slate-600 font-bold hover:bg-slate-100 rounded-xl"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={isCreatingClass}
                  className="px-4 py-2 bg-indigo-600 hover:bg-indigo-700 text-white font-bold rounded-xl shadow-sm"
                >
                  {isCreatingClass ? 'Creating...' : 'Create Cohort'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* DETAILED STUDENT REPORT MODAL */}
      {inspectingSession && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fadeIn overflow-y-auto">
          <div className="bg-white border border-slate-200 rounded-3xl max-w-4xl w-full p-6 sm:p-8 shadow-2xl space-y-6 my-8 max-h-[90vh] overflow-y-auto text-slate-800">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-slate-200 pb-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold px-2 py-0.5 rounded bg-sky-50 text-sky-700 border border-sky-200">
                    Comprehensive Student Report
                  </span>
                  <span className="text-xs text-slate-500 font-mono">Cohort: {inspectingSession.classroom_name || 'Group Alpha'}</span>
                </div>
                <h3 className="text-xl font-bold text-slate-900 mt-1">
                  {inspectingSession.student_name} <span className="text-slate-500 text-sm font-normal">({inspectingSession.student_id})</span>
                </h3>
                <p className="text-xs text-slate-500">Problem Task: {inspectingSession.title}</p>
              </div>

              <button
                onClick={() => setInspectingSession(null)}
                className="text-slate-400 hover:text-slate-700 text-2xl font-bold px-2"
              >
                &times;
              </button>
            </div>

            {isLoadingReport ? (
              <div className="py-12 text-center text-xs text-slate-500 font-mono animate-pulse">
                Loading student report, AST metrics, and viva evaluations...
              </div>
            ) : (
              <div className="space-y-6">
                {/* AI Classification & Confidence */}
                <div className="p-4 rounded-2xl bg-slate-50 border border-slate-200 flex flex-wrap items-center justify-between gap-4">
                  <div>
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block mb-1 font-bold">
                      System Comprehension Verdict
                    </span>
                    <div className="flex items-center gap-2">
                      <span className={`px-3 py-1 rounded-full text-xs font-black border ${
                        (studentReport?.overall_label || inspectingSession.latest_verdict) === 'Understood'
                          ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                          : (studentReport?.overall_label || inspectingSession.latest_verdict) === 'Partial Understanding'
                          ? 'bg-amber-50 text-amber-800 border-amber-300'
                          : (studentReport?.overall_label || inspectingSession.latest_verdict) === 'Pending / Insufficient Evidence'
                          ? 'bg-slate-100 text-slate-800 border-slate-300'
                          : 'bg-rose-50 text-rose-800 border-rose-300'
                      }`}>
                        {studentReport?.overall_label || inspectingSession.latest_verdict || 'Understood'}
                      </span>
                      <span className="text-xs text-slate-600 font-mono font-medium">
                        Model Confidence: <strong className="text-sky-700">{studentReport ? Math.round(studentReport.confidence_score * 100) : 85}%</strong>
                      </span>
                    </div>
                  </div>

                  <div className="text-right">
                    <span className="text-[10px] font-mono uppercase tracking-wider text-slate-500 block mb-1 font-bold">
                      Viva Questions Answered
                    </span>
                    <span className="text-sm font-bold text-slate-900 font-mono">
                      {studentQuestions.filter((q) => q.has_answer || !!q.assessment).length} / {studentQuestions.length} Questions
                    </span>
                  </div>
                </div>

                {/* Evidence Summary */}
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 space-y-1">
                  <span className="font-bold text-sky-800 uppercase tracking-wider text-[10px]">
                    Synthesized Evidence Summary:
                  </span>
                  <p className="leading-relaxed">
                    {studentReport?.evidence_summary || 'The student demonstrated consistent control-flow reasoning across iterations, with active revision patterns and valid AST logic.'}
                  </p>
                </div>

                {/* Question by Question Viva Breakdown */}
                <div className="space-y-3">
                  <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                    AI Viva Questions & Student Explanations
                  </h4>

                  {studentQuestions.length === 0 ? (
                    <p className="text-xs text-slate-400 italic p-4 bg-slate-50 rounded-xl">
                      No viva questions generated yet for this student.
                    </p>
                  ) : (
                    studentQuestions.map((q, idx) => (
                      <div key={q.id} className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2 text-xs">
                        <div className="flex items-center justify-between">
                          <span className="font-mono font-bold text-indigo-700">Q{idx + 1}: {q.concept}</span>
                          <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-slate-200 text-slate-700">
                            {q.difficulty}
                          </span>
                        </div>
                        <p className="font-medium text-slate-900">{q.question_text}</p>
                        
                        {q.assessment ? (
                          <div className="mt-2 pt-2 border-t border-slate-200 space-y-1.5">
                            <div className="flex items-center gap-2">
                              <span className="font-bold text-slate-500 text-[10px] uppercase">Evaluation:</span>
                              <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                                q.assessment.evaluation_label === 'Correct'
                                  ? 'bg-emerald-50 text-emerald-800 border border-emerald-200'
                                  : q.assessment.evaluation_label === 'Partially Correct'
                                  ? 'bg-amber-50 text-amber-800 border border-amber-200'
                                  : 'bg-rose-50 text-rose-800 border border-rose-200'
                              }`}>
                                {q.assessment.evaluation_label}
                              </span>
                              <span className="text-slate-500 text-[10px]">Score: {Math.round((q.assessment.score ?? 0) * 100)}%</span>
                            </div>
                            <p className="text-slate-700 italic bg-white p-2.5 rounded-lg border border-slate-200">
                              "{q.assessment.student_answer}"
                            </p>
                            <p className="text-slate-500 text-[11px]">
                              <strong>Rationale:</strong> {q.assessment.feedback}
                            </p>
                          </div>
                        ) : (
                          <p className="text-slate-400 italic text-[11px] pt-1">
                            Awaiting student response.
                          </p>
                        )}
                      </div>
                    ))
                  )}
                </div>

                {/* Submitted Code Preview */}
                {studentSubmissions.length > 0 && (
                  <div className="space-y-2">
                    <h4 className="text-xs font-bold uppercase tracking-wider text-slate-700 font-mono">
                      Latest Submitted Code
                    </h4>
                    <pre className="bg-slate-50 p-4 rounded-xl border border-slate-200 font-mono text-[11px] text-slate-900 overflow-x-auto max-h-48 leading-relaxed">
                      {studentSubmissions[0].code}
                    </pre>
                  </div>
                )}

                {/* Process Telemetry & Problem-Solving Timeline */}
                <div className="space-y-2">
                  <ProcessTimeline features={studentFeatures} />
                </div>

                {/* Instructor Ground-Truth Annotation Form */}
                <div className="p-5 rounded-2xl bg-indigo-50/50 border border-indigo-200 space-y-4">
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="w-5 h-5 text-indigo-700" />
                    <h4 className="font-bold text-indigo-950 text-sm">
                      Instructor Ground-Truth Review & Verification
                    </h4>
                  </div>

                  {reviewSuccess && (
                    <div className="p-3 bg-emerald-100 border border-emerald-300 text-emerald-900 rounded-xl text-xs flex items-center gap-2 font-semibold">
                      <CheckCircle2 className="w-4 h-4 text-emerald-700" />
                      <span>Review submitted! This record will serve as ground-truth for scientific evaluation.</span>
                    </div>
                  )}

                  <form onSubmit={handleSubmitInstructorReview} className="space-y-3 text-xs">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                          Final Instructor Verdict
                        </label>
                        <select
                          value={assignedLabel}
                          onChange={(e) => setAssignedLabel(e.target.value)}
                          className="w-full bg-white border border-slate-300 rounded-xl px-3 py-2 text-slate-900 font-semibold focus:outline-none focus:ring-2 focus:ring-indigo-500"
                        >
                          <option value="Understood">Understood (Solid conceptual mastery)</option>
                          <option value="Partial Understanding">Partial Understanding (Gaps present)</option>
                          <option value="Possible Misconception">Possible Misconception (Definite flaw detected)</option>
                        </select>
                      </div>

                      <div>
                        <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                          Agreement with AI Assessment
                        </label>
                        <div className="flex items-center gap-3 mt-2">
                          <label className="inline-flex items-center gap-1.5 cursor-pointer">
                            <input
                              type="radio"
                              name="agree"
                              checked={agreeWithAi === true}
                              onChange={() => setAgreeWithAi(true)}
                              className="text-indigo-600"
                            />
                            <span>Agree with AI</span>
                          </label>
                          <label className="inline-flex items-center gap-1.5 cursor-pointer">
                            <input
                              type="radio"
                              name="agree"
                              checked={agreeWithAi === false}
                              onChange={() => setAgreeWithAi(false)}
                              className="text-indigo-600"
                            />
                            <span>Override AI</span>
                          </label>
                        </div>
                      </div>
                    </div>

                    <div>
                      <label className="block font-mono font-semibold uppercase text-slate-700 mb-1">
                        Supervisor Qualitative Notes
                      </label>
                      <textarea
                        rows={2}
                        value={feedbackNotes}
                        onChange={(e) => setFeedbackNotes(e.target.value)}
                        placeholder="Add pedagogical feedback, specific code misconceptions identified, or justification for override..."
                        className="w-full bg-white border border-slate-300 rounded-xl p-2.5 text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-indigo-500"
                      />
                    </div>

                    <div className="flex items-center justify-end gap-3 pt-1">
                      <button
                        type="button"
                        onClick={() => setInspectingSession(null)}
                        className="px-4 py-2 text-slate-600 hover:text-slate-900 font-semibold"
                      >
                        Close
                      </button>
                      <button
                        type="submit"
                        disabled={isSubmittingReview}
                        className="px-5 py-2.5 rounded-xl font-bold bg-indigo-600 hover:bg-indigo-700 text-white shadow-md transition-all disabled:opacity-50"
                      >
                        {isSubmittingReview ? 'Submitting...' : 'Save Ground-Truth Review'}
                      </button>
                    </div>
                  </form>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
