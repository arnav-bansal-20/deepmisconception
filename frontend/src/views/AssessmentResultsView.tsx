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
import { CodingSession, ComprehensionReport, ProcessFeatures } from '../types';
import { api } from '../services/api';
import { ProcessTimeline } from '../components/ProcessTimeline';

interface AssessmentResultsViewProps {
  session: CodingSession | null;
  onNavigateToWorkspace: () => void;
}

export const AssessmentResultsView: React.FC<AssessmentResultsViewProps> = ({
  session,
  onNavigateToWorkspace,
}) => {
  const [report, setReport] = useState<ComprehensionReport | null>(null);
  const [features, setFeatures] = useState<ProcessFeatures | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isRegenerating, setIsRegenerating] = useState(false);

  const loadData = async (forceGenerate: boolean = false) => {
    if (!session) return;
    setIsLoading(true);
    try {
      if (forceGenerate) {
        setIsRegenerating(true);
        await api.generateReport(session.id);
      }
      const [rep, feat] = await Promise.allSettled([
        api.getSessionReport(session.id),
        api.getProcessFeatures(session.id),
      ]);
      if (rep.status === 'fulfilled') setReport(rep.value);
      if (feat.status === 'fulfilled') setFeatures(feat.value);
    } catch (err) {
      console.error('Failed to load assessment report', err);
    } finally {
      setIsLoading(false);
      setIsRegenerating(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [session]);

  if (!session) {
    return (
      <div className="text-center py-20 text-slate-500">
        No active session selected.
      </div>
    );
  }

  // If no report generated yet, show prompt with generate button
  if (!report && !isLoading) {
    return (
      <div className="max-w-2xl mx-auto py-16 px-4 text-center space-y-4">
        <div className="w-16 h-16 rounded-2xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto shadow-sm">
          <Brain className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-slate-900">Comprehension Report Ready to Synthesize</h3>
        <p className="text-sm text-slate-600 leading-relaxed">
          Synthesize a multi-modal report combining your code AST structure, isolated execution output, viva reasoning, and editor telemetry.
        </p>
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
            <span>Return to Workspace</span>
            <ChevronRight className="w-4 h-4" />
          </button>
        </div>
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
    <div className="max-w-6xl mx-auto space-y-8 px-4 sm:px-6 py-6">
      {/* Top Banner */}
      <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-bold uppercase">
                Diagnosis Report
              </span>
              <span className="text-xs text-slate-500 font-mono">
                Cohort: {session.classroom_name || 'Group Alpha'} &bull; ID: {session.id.slice(0, 8)}...
              </span>
            </div>
            <h2 className="text-2xl font-black text-slate-900 mt-1">
              Comprehension & Misconception Evaluation
            </h2>
            <p className="text-xs text-slate-500">
              Evaluated Candidate: <strong className="text-slate-800">{session.student_name}</strong> ({session.student_id})
            </p>
          </div>

          {/* Classification Badge & Re-evaluate */}
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

        {/* Flagged for Review Notice (if conflicting signals) */}
        {report?.needs_instructor_review && (
          <div className="p-3.5 bg-amber-50 border border-amber-300 text-amber-900 rounded-xl text-xs flex items-start gap-2.5">
            <AlertTriangle className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" />
            <div>
              <strong className="block font-bold">Uncertainty Flagged for Instructor Review:</strong>
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
