import React, { useState, useEffect } from 'react';
import { 
  Award, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  HelpCircle,
  ShieldCheck, 
  Activity, 
  Brain, 
  Layers, 
  FileText,
  Clock,
  Sparkles,
  RefreshCw,
  ChevronRight,
  BookOpen,
  ArrowRight,
  Code2,
  Terminal,
  Compass
} from 'lucide-react';
import { CodingSession, ComprehensionReport, ProcessFeatures, AuthUser } from '../types';
import { api } from '../services/api';
import { ProcessTimeline } from '../components/ProcessTimeline';

interface AssessmentResultsViewProps {
  session: CodingSession | null;
  currentUser?: AuthUser | null;
  onNavigateToWorkspace: () => void;
  onSelectSession?: (session: CodingSession) => void;
}

export const AssessmentResultsView: React.FC<AssessmentResultsViewProps> = ({
  session,
  currentUser,
  onNavigateToWorkspace,
  onSelectSession,
}) => {
  const [selectedSession, setSelectedSession] = useState<CodingSession | null>(session);
  const [report, setReport] = useState<ComprehensionReport | null>(null);
  const [features, setFeatures] = useState<ProcessFeatures | null>(null);
  const [studentSessions, setStudentSessions] = useState<CodingSession[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);

  // Keep selectedSession synced with incoming session prop if present
  useEffect(() => {
    if (session) {
      setSelectedSession(session);
    }
  }, [session?.id]);

  // Fetch all sessions belonging to the student
  useEffect(() => {
    if (currentUser?.id) {
      api.listSessions({ student_id: currentUser.id })
        .then((list) => {
          setStudentSessions(list);
          if (!selectedSession && list.length > 0) {
            setSelectedSession(list[0]);
            if (onSelectSession) onSelectSession(list[0]);
          }
        })
        .catch((e) => console.error('Failed loading student sessions for report:', e));
    }
  }, [currentUser]);

  const loadData = async (forceGenerate: boolean = false, targetSessionId?: string) => {
    const targetId = targetSessionId || selectedSession?.id || session?.id;
    if (!targetId) return;
    setIsLoading(true);
    try {
      if (forceGenerate) {
        setIsRegenerating(true);
        await api.generateReport(targetId);
      }
      const [rep, feat] = await Promise.allSettled([
        api.getSessionReport(targetId),
        api.getProcessFeatures(targetId),
      ]);
      if (rep.status === 'fulfilled') setReport(rep.value);
      else setReport(null);
      if (feat.status === 'fulfilled') setFeatures(feat.value);
      else setFeatures(null);
    } catch (err) {
      console.error('Failed to load assessment report', err);
      setReport(null);
    } finally {
      setIsLoading(false);
      setIsRegenerating(false);
    }
  };

  useEffect(() => {
    if (selectedSession?.id) {
      loadData(false, selectedSession.id);
    }
  }, [selectedSession?.id]);

  const handleSelectProblem = (s: CodingSession) => {
    setSelectedSession(s);
    if (onSelectSession) onSelectSession(s);
    loadData(false, s.id);
  };

  const activeSession = selectedSession || session;

  if (!activeSession && studentSessions.length === 0) {
    return (
      <div className="max-w-md mx-auto py-20 px-4 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-sky-50 text-sky-600 border border-sky-200 flex items-center justify-center mx-auto">
          <Award className="w-6 h-6" />
        </div>
        <h3 className="text-lg font-bold text-slate-900">No Assessment Reports Yet</h3>
        <p className="text-xs text-slate-500 leading-relaxed">
          Complete a coding task and answer its AI viva questions in the Workspace to generate your problem diagnosis report.
        </p>
        <button
          onClick={onNavigateToWorkspace}
          className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs transition-colors shadow-sm"
        >
          <span>Go to Workspace</span>
          <ChevronRight className="w-4 h-4" />
        </button>
      </div>
    );
  }

  const verdict = report?.overall_label || 'Partial Understanding';
  const confidence = report ? Math.round(report.confidence_score * 100) : 85;
  const scores = report?.multimodal_scores || {
    ast_structural_score: 0.85,
    execution_fidelity_score: 1.0,
    viva_reasoning_score: 0.80,
    process_authenticity_score: 0.90,
  };

  return (
    <div className="max-w-6xl mx-auto space-y-6 px-4 sm:px-6 py-6">
      {/* Problem Selection Cards */}
      <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-sky-600" />
              <span>Select Problem To View Report</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any problem below to inspect its AI comprehension and misconception diagnosis report.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200 font-semibold">
            {studentSessions.length} Problem{studentSessions.length === 1 ? '' : 's'} Solved
          </span>
        </div>

        {/* Problem-wise Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-1">
          {studentSessions.map((s) => {
            const isSelected = activeSession?.id === s.id;
            return (
              <div
                key={s.id}
                onClick={() => handleSelectProblem(s)}
                className={`p-4 rounded-xl cursor-pointer transition-all border text-left flex flex-col justify-between ${
                  isSelected
                    ? 'bg-sky-50/70 border-sky-500 ring-2 ring-sky-500/20 shadow-sm'
                    : 'bg-slate-50/60 hover:bg-white border-slate-200 hover:border-slate-300 hover:shadow-sm'
                }`}
              >
                <div>
                  <div className="flex items-center justify-between gap-2 mb-2">
                    <span className="text-[10px] font-mono uppercase px-2 py-0.5 rounded font-bold bg-white border border-slate-200 text-slate-700">
                      {s.language}
                    </span>
                    <div className="flex items-center gap-1.5 flex-wrap justify-end">
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold capitalize border ${
                        s.status === 'completed'
                          ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                          : s.status === 'reviewed'
                          ? 'bg-indigo-50 text-indigo-800 border-indigo-200'
                          : 'bg-amber-50 text-amber-800 border-amber-300'
                      }`}>
                        {s.status}
                      </span>
                      {s.latest_verdict && (
                        <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                          s.latest_verdict === 'Understood'
                            ? 'bg-emerald-50 text-emerald-800 border border-emerald-300'
                            : s.latest_verdict === 'Partial Understanding'
                            ? 'bg-amber-50 text-amber-800 border border-amber-300'
                            : 'bg-rose-50 text-rose-800 border border-rose-300'
                        }`}>
                          {s.latest_verdict}
                        </span>
                      )}
                    </div>
                  </div>
                  <h4 className={`text-xs font-bold leading-snug line-clamp-2 ${isSelected ? 'text-sky-950 font-black' : 'text-slate-800'}`}>
                    {s.title}
                  </h4>
                </div>

                <div className="pt-2.5 mt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span>{new Date(s.created_at).toLocaleDateString()}</span>
                  <span className={`font-bold flex items-center gap-1 ${isSelected ? 'text-sky-700 font-bold' : 'text-slate-400'}`}>
                    {isSelected ? 'Viewing Report ✓' : 'Click to View →'}
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Selected Problem Report Section */}
      {isLoading ? (
        <div className="bg-white border border-slate-200 p-12 rounded-2xl text-center space-y-3 shadow-sm">
          <RefreshCw className="w-8 h-8 text-sky-600 animate-spin mx-auto" />
          <p className="text-xs font-mono text-slate-500 font-semibold">Loading diagnosis report for "{activeSession?.title}"...</p>
        </div>
      ) : !report ? (
        <div className="bg-white border border-slate-200 p-8 rounded-2xl text-center space-y-4 shadow-sm max-w-xl mx-auto">
          <div className="w-12 h-12 rounded-2xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto shadow-sm">
            <Brain className="w-6 h-6" />
          </div>
          <div>
            <h3 className="text-base font-bold text-slate-900">
              Report Ready to Synthesize for "{activeSession?.title}"
            </h3>
            <p className="text-xs text-slate-500 mt-1 max-w-md mx-auto leading-relaxed">
              Synthesize a multi-modal evaluation fusing your code structure, isolated execution traces, viva answer explanations, and process telemetry.
            </p>
          </div>
          <div className="pt-2 flex items-center justify-center gap-3">
            <button
              onClick={() => loadData(true)}
              disabled={isRegenerating}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs transition-all shadow-md disabled:opacity-50"
            >
              <Sparkles className="w-4 h-4" />
              <span>{isRegenerating ? 'Synthesizing Report...' : 'Synthesize Report Now'}</span>
            </button>
            <button
              onClick={onNavigateToWorkspace}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs transition-colors"
            >
              <span>Open in Workspace</span>
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>
        </div>
      ) : (
        /* Full Multimodal Comprehension Report */
        <div className="space-y-6">
          {/* Top Banner */}
          <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm space-y-4">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-bold uppercase">
                    Diagnosis Report
                  </span>
                  <span className="text-xs text-slate-500 font-mono">
                    Problem: <strong className="text-slate-800">{activeSession?.title}</strong> &bull; Group: {activeSession?.classroom_name || 'Group Alpha'}
                  </span>
                </div>
                <h2 className="text-2xl font-black text-slate-900 mt-1">
                  Comprehension & Misconception Evaluation
                </h2>
                <p className="text-xs text-slate-500 mt-1">
                  Candidate: <strong className="text-slate-800">{activeSession?.student_name}</strong> ({activeSession?.student_id}) &bull; Evaluated {new Date(activeSession?.updated_at || Date.now()).toLocaleDateString()}
                </p>
              </div>

              {/* Classification Badge & Refresh */}
              <div className="flex items-center gap-4">
                <button
                  onClick={() => loadData(true)}
                  disabled={isRegenerating}
                  className="p-2 text-slate-500 hover:text-slate-800 border border-slate-200 rounded-xl hover:bg-slate-50 transition-colors text-xs flex items-center gap-1.5"
                  title="Re-synthesize latest report"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isRegenerating ? 'animate-spin' : ''}`} />
                  <span className="hidden sm:inline">Refresh Diagnosis</span>
                </button>

                <div className="text-right">
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-xl text-sm font-black border shadow-sm ${
                    verdict === 'Understood'
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                      : verdict === 'Partial Understanding'
                      ? 'bg-amber-50 text-amber-800 border-amber-300'
                      : verdict === 'Pending / Insufficient Evidence'
                      ? 'bg-slate-100 text-slate-800 border-slate-300'
                      : 'bg-rose-50 text-rose-800 border-rose-300'
                  }`}>
                    {verdict === 'Understood' && <CheckCircle2 className="w-5 h-5 text-emerald-600" />}
                    {verdict === 'Partial Understanding' && <AlertTriangle className="w-5 h-5 text-amber-600" />}
                    {verdict === 'Pending / Insufficient Evidence' && <HelpCircle className="w-5 h-5 text-slate-600" />}
                    {verdict === 'Possible Misconception' && <XCircle className="w-5 h-5 text-rose-600" />}
                    <span>{verdict}</span>
                  </div>
                  <div className="text-[11px] font-mono text-slate-500 mt-1 font-medium">
                    Model Confidence: <span className="text-sky-700 font-bold">{confidence}%</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Flagged for Review Notice */}
            {report?.needs_instructor_review && (
              <div className="p-3.5 bg-amber-50 border border-amber-300 text-amber-900 rounded-xl text-xs flex items-start gap-2.5">
                <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
                <div>
                  <strong className="block font-bold">Uncertainty Flagged for Instructor Review:</strong>
                  <p className="mt-0.5 text-amber-800 leading-relaxed">
                    The model flagged potential ambiguity or conflicting signals between code execution and viva explanations. An instructor will review this session.
                  </p>
                </div>
              </div>
            )}
          </div>

          {/* Multimodal Score Profiles */}
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">AST Structure</span>
                <Code2 className="w-4 h-4 text-sky-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {Math.round((scores.ast_structural_score || 0) * 100)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Loop &amp; recursion syntax logic</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Execution</span>
                <Terminal className="w-4 h-4 text-emerald-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {Math.round((scores.execution_fidelity_score || 0) * 100)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Sandbox test fidelity</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Viva Reasoning</span>
                <Brain className="w-4 h-4 text-indigo-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {Math.round((scores.viva_reasoning_score || 0) * 100)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Conceptual alignment</div>
            </div>

            <div className="bg-white border border-slate-200 p-4 rounded-xl shadow-sm">
              <div className="flex items-center justify-between text-slate-500 mb-1">
                <span className="text-xs font-bold uppercase tracking-wider">Process Authenticity</span>
                <Compass className="w-4 h-4 text-violet-600" />
              </div>
              <div className="text-2xl font-bold text-slate-900">
                {Math.round((scores.process_authenticity_score || 0) * 100)}%
              </div>
              <div className="text-[10px] text-slate-500 mt-1">Incremental edit trajectory</div>
            </div>
          </div>

          {/* Evidence Summary & Identified Misconceptions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <FileText className="w-4 h-4 text-sky-600" />
                Evidence &amp; Synthesis Summary
              </h3>
              <p className="text-xs text-slate-700 leading-relaxed font-sans bg-slate-50 p-4 rounded-xl border border-slate-200">
                {report.evidence_summary || 'The student demonstrated consistent reasoning with valid code syntax and aligned explanation.'}
              </p>
            </div>

            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-600" />
                Detected Misconceptions
              </h3>
              {report.detected_misconceptions && report.detected_misconceptions.length > 0 ? (
                <ul className="space-y-2">
                  {report.detected_misconceptions.map((m, i) => (
                    <li key={i} className="text-xs bg-rose-50 text-rose-900 p-3 rounded-xl border border-rose-200 flex items-start gap-2">
                      <span className="font-bold text-rose-600">•</span>
                      <span>{m}</span>
                    </li>
                  ))}
                </ul>
              ) : (
                <div className="text-xs text-emerald-800 bg-emerald-50 p-4 rounded-xl border border-emerald-200 flex items-center gap-2">
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                  <span>No severe conceptual misconceptions detected for this problem.</span>
                </div>
              )}
            </div>
          </div>

          {/* Process Timeline */}
          {features && (
            <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3">
              <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                <Clock className="w-4 h-4 text-indigo-600" />
                Development Trajectory &amp; Telemetry
              </h3>
              <ProcessTimeline features={features} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
              <span>
                Conflicting evidence was detected between code structural complexity and verbal viva explanations. The report has been flagged for faculty audit in the Instructor Studio.
              </span>
            </div>
          </div>
        )}

        {/* Evidence Summary Box */}
        <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 leading-relaxed space-y-1">
          <span className="font-bold text-sky-800 uppercase tracking-wider block">
            Synthesized Evidence Summary:
          </span>
          <p>
            {report?.evidence_summary || 
              'The student demonstrated understanding of loop sequencing and variable accumulation, with valid AST structure and consistent viva justifications.'}
          </p>
        </div>
      </div>

      {/* Multimodal 4-Dimension Evidence Scorecard */}
      <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
        <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
          <Brain className="w-4 h-4 text-indigo-600" />
          <span>Multimodal Evidence Fusion Dimensions</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-700 flex items-center gap-1.5">
                <Code2 className="w-3.5 h-3.5 text-sky-600" />
                <span>AST Structure</span>
              </span>
              <span className="font-mono font-bold text-sky-700">{Math.round(scores.ast_structural_score * 100)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-sky-500 rounded-full transition-all duration-500" 
                style={{ width: `${scores.ast_structural_score * 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-500">Syntax validity, loop constructs, function definitions</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-700 flex items-center gap-1.5">
                <Terminal className="w-3.5 h-3.5 text-emerald-600" />
                <span>Execution Sandbox</span>
              </span>
              <span className="font-mono font-bold text-emerald-700">{Math.round(scores.execution_fidelity_score * 100)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-emerald-500 rounded-full transition-all duration-500" 
                style={{ width: `${scores.execution_fidelity_score * 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-500">Zero runtime errors, correct output generation</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-700 flex items-center gap-1.5">
                <Brain className="w-3.5 h-3.5 text-indigo-600" />
                <span>Viva Reasoning</span>
              </span>
              <span className="font-mono font-bold text-indigo-700">{Math.round(scores.viva_reasoning_score * 100)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-indigo-500 rounded-full transition-all duration-500" 
                style={{ width: `${scores.viva_reasoning_score * 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-500">Conceptual depth, factual code alignment</p>
          </div>

          <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 space-y-2">
            <div className="flex items-center justify-between text-xs">
              <span className="font-bold text-slate-700 flex items-center gap-1.5">
                <Activity className="w-3.5 h-3.5 text-teal-600" />
                <span>Process Development (20%)</span>
              </span>
              <span className="font-mono font-bold text-teal-700">{Math.round(((scores as any).process_overall_score ?? scores.process_rhythm_score ?? scores.process_authenticity_score ?? 0.75) * 100)}%</span>
            </div>
            <div className="w-full h-2 bg-slate-200 rounded-full overflow-hidden">
              <div 
                className="h-full bg-teal-500 rounded-full transition-all duration-500" 
                style={{ width: `${((scores as any).process_overall_score ?? scores.process_rhythm_score ?? scores.process_authenticity_score ?? 0.75) * 100}%` }}
              />
            </div>
            <p className="text-[10px] text-slate-500">Incremental edits, error recovery, revision cycles, testing</p>
          </div>
        </div>
      </div>

      {/* Grid: Diagnosed Misconceptions & Recommended Interventions */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* Diagnosed Misconceptions Card */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <Layers className="w-4 h-4 text-indigo-600" />
            <span>Diagnosed Misconceptions</span>
          </h3>

          <div className="space-y-2.5">
            {report?.detected_misconceptions && report.detected_misconceptions.length > 0 ? (
              report.detected_misconceptions.map((m, idx) => (
                <div key={idx} className="p-3.5 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2.5">
                  <AlertTriangle className="w-4 h-4 text-rose-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <strong className="block text-rose-950 font-bold">Identified Misconception:</strong>
                    <span className="text-rose-900 font-medium">{m}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="p-4 rounded-xl bg-emerald-50 border border-emerald-200 text-xs text-emerald-800 flex items-center gap-2 font-medium">
                <CheckCircle2 className="w-4 h-4 text-emerald-600 flex-shrink-0" />
                <span>No severe conceptual misconceptions identified in viva explanations or code execution.</span>
              </div>
            )}
          </div>
        </div>

        {/* Recommended Pedagogical Interventions */}
        <div className="bg-white border border-slate-200 rounded-2xl p-6 shadow-sm space-y-4">
          <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
            <Compass className="w-4 h-4 text-sky-600" />
            <span>Recommended Next Steps & Exercises</span>
          </h3>

          <div className="space-y-2.5">
            {report?.recommended_interventions && report.recommended_interventions.length > 0 ? (
              report.recommended_interventions.map((item, idx) => (
                <div key={idx} className="p-3.5 rounded-xl bg-sky-50 border border-sky-200 text-xs text-sky-900 flex items-start gap-2.5">
                  <BookOpen className="w-4 h-4 text-sky-600 flex-shrink-0 mt-0.5" />
                  <div>
                    <span className="font-semibold">{item}</span>
                  </div>
                </div>
              ))
            ) : (
              <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-600">
                Continue solving targeted programming problems to reinforce algorithmic invariants.
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Embedded Process Telemetry Timeline */}
      <ProcessTimeline features={features} />
    </div>
  );
};
