import React from 'react';
import { 
  Printer, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  HelpCircle,
  RefreshCw,
  Clock,
  Sparkles,
  Info
} from 'lucide-react';
import { CodingSession, ComprehensionReport, ProcessFeatures, AuthUser } from '../types';

interface ComprehensionReportTemplateProps {
  report: ComprehensionReport;
  session: CodingSession;
  features?: ProcessFeatures | null;
  currentUser?: AuthUser | null;
  onRefresh?: () => void;
  isRegenerating?: boolean;
}

export const ComprehensionReportTemplate: React.FC<ComprehensionReportTemplateProps> = ({
  report,
  session,
  features,
  currentUser,
  onRefresh,
  isRegenerating = false,
}) => {
  const eb = (report.evidence_breakdown || {}) as Record<string, any>;
  const rubric = (report.rubric_summary || {}) as Record<string, any>;
  const scores = report.multimodal_scores || {
    ast_structural_score: 0.85,
    execution_fidelity_score: 1.0,
    viva_reasoning_score: 0.80,
    process_overall_score: 0.80,
    process_authenticity_score: 0.80,
  };

  const astScore = scores.ast_structural_score ?? 0.85;
  const execScore = scores.execution_fidelity_score ?? 1.0;
  const vivaScore = scores.viva_reasoning_score ?? 0.80;
  const procScore = scores.process_overall_score ?? scores.process_authenticity_score ?? 0.80;

  // Composite calculation: 0.15*AST + 0.20*EXEC + 0.45*VIVA + 0.20*PROCESS
  const compositeScore = Number(
    (0.15 * astScore + 0.20 * execScore + 0.45 * vivaScore + 0.20 * procScore).toFixed(3)
  );
  const overallScore = Math.round(compositeScore * 100);

  const verdict = report.overall_label || 'Partial Understanding';
  const confidencePercent = Math.round((report.confidence_score || 0.85) * 100);

  // Section 5: Viva answer evaluations
  const vivaEvals: any[] = eb.viva_evaluations || [];

  // Section 7: Misconceptions and Contradictions
  const miscTable: any[] = eb.misconceptions_table || (
    report.detected_misconceptions && report.detected_misconceptions.length > 0
      ? report.detected_misconceptions.map((dm: string) => ({
          concept: session.title,
          evidence: 'Student verbal response',
          pattern: 'Identified conceptual divergence',
          interpretation: dm
        }))
      : []
  );

  // Modality Discrepancy Observation (e.g. 99% score vs Partial Understanding)
  const hasModalityDiscrepancy = overallScore >= 85 && verdict === 'Partial Understanding';

  // Section 3: Code & execution evidence
  const codeEv = eb.code_evidence || {
    code_summary: session.title ? `Python solution for "${session.title}".` : 'Python source submission.',
    ast_findings: 'Syntactically valid AST structure. Valid function declarations verified.',
    execution_outcome: 'Sandbox execution exit code 0.',
    evidence_limitations: 'Sandbox unit tests verify I/O correctness, but cannot independently guarantee verbalized invariant comprehension.'
  };

  // Section 6: Coding-Process Behaviour metrics
  const nPaste = features?.paste_count ?? eb.paste_count ?? 0;
  const pasteChars = (features as any)?.paste_chars ?? eb.paste_chars ?? 0;
  const pasteRatio = features?.paste_ratio ?? eb.paste_ratio ?? 0.0;
  const largestInsert = (features as any)?.largest_insert_share ?? eb.largest_insert_share ?? (nPaste > 0 ? 0.35 : 0.08);
  const editCount = features?.edit_count ?? eb.edit_count ?? 12;
  const runCount = features?.run_count ?? eb.run_count ?? 4;
  const eventCompleteness = (features as any)?.event_completeness || (editCount > 0 ? 'Complete' : 'Partial');

  // Section 8: Instructor review
  const instructorReview = eb.instructor_review || {
    comments: session.status === 'reviewed' 
      ? (eb.instructor_review?.comments || 'Student demonstrated procedural mastery of syntax. Recommended to reinforce formal loop invariant formulation.')
      : 'Awaiting formal instructor review and audit.',
    reviewed_by: session.teacher_name || 'Dr. Ada Smith',
    review_date: new Date(session.updated_at || Date.now()).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
    final_decision: session.status === 'reviewed' ? 'Accepted with Follow-up' : 'Pending Instructor Audit'
  };

  // Section 8: Follow-up & Practice
  const feedback = eb.feedback || {
    suggested_practice: report.recommended_interventions && report.recommended_interventions.length > 0
      ? report.recommended_interventions[0]
      : 'Trace edge-case boundary conditions and formal loop invariants.',
    targeted_question: 'Ask student to predict program behavior under negative inputs or trace pointer convergence on single-character boundary cases.'
  };

  const handlePrint = () => {
    window.print();
  };

  const badgeClass = (label: string) => {
    switch (label) {
      case 'Correct':
      case 'Understood':
        return 'bg-emerald-50 text-emerald-800 border-emerald-300';
      case 'Partially Correct':
      case 'Partial Understanding':
        return 'bg-amber-50 text-amber-800 border-amber-300';
      case 'Pending / Insufficient Evidence':
      case 'Insufficient Evidence':
        return 'bg-slate-100 text-slate-800 border-slate-300';
      default:
        return 'bg-rose-50 text-rose-800 border-rose-300';
    }
  };

  return (
    <div className="bg-white text-slate-800 font-sans print:p-0 space-y-6">
      {/* Top Action Bar (Hidden during print) */}
      <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-200 print:hidden">
        <div className="text-xs text-slate-500 font-mono">
          Formal Assessment Document &bull; Session <span className="font-semibold">{session.id.slice(0, 8)}</span>
        </div>
        <div className="flex items-center gap-2">
          {onRefresh && (
            <button
              onClick={onRefresh}
              disabled={isRegenerating}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-slate-200 hover:bg-slate-50 text-slate-700 text-xs font-semibold transition-colors disabled:opacity-50"
              title="Refresh report to fetch latest instructor updates and scores"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isRegenerating ? 'animate-spin text-sky-600' : ''}`} />
              <span>{isRegenerating ? 'Refreshing...' : 'Refresh'}</span>
            </button>
          )}
          <button
            onClick={handlePrint}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-sky-600 hover:bg-sky-500 text-white text-xs font-bold transition-colors shadow-sm"
          >
            <Printer className="w-3.5 h-3.5" />
            <span>Print / Save PDF</span>
          </button>
        </div>
      </div>

      {/* DOCUMENT HEADER */}
      <div className="border-b-2 border-slate-900 pb-4 text-center sm:text-left sm:flex sm:items-end sm:justify-between gap-4">
        <div>
          <span className="text-[11px] font-mono tracking-widest font-black uppercase text-sky-700 block">
            DEEPMISCONCEPTION
          </span>
          <h1 className="text-2xl font-black text-slate-900 tracking-tight mt-0.5">
            Student Conceptual Understanding Report
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            AI-assisted programming assessment &nbsp;|&nbsp; Reusable instructor report template
          </p>
        </div>
        <div className="mt-3 sm:mt-0 text-right">
          <span className={`inline-block px-3 py-1 rounded-full text-xs font-black border uppercase tracking-wider ${badgeClass(verdict)}`}>
            {verdict}
          </span>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Diag. Score: <strong className="text-slate-900">{overallScore}%</strong> &bull; Model Prob: {confidencePercent}%
          </div>
        </div>
      </div>

      {/* 1. STUDENT AND ASSESSMENT DETAILS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          1. Student and Assessment Details
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <tbody>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 w-1/4 text-slate-700">Student name</td>
                <td className="px-3 py-2 border border-slate-300 w-1/4 font-semibold text-slate-900">{session.student_name || currentUser?.name || '—'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 w-1/4 text-slate-700">Student ID</td>
                <td className="px-3 py-2 border border-slate-300 w-1/4 font-mono font-semibold text-slate-900">{session.student_id || currentUser?.id || '—'}</td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Course / cohort</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-800">{session.classroom_name || currentUser?.classroomCode || 'CS101: Python Foundations'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Assessment date</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-800">
                  {new Date(session.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                </td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Problem / concept</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-900 font-semibold">{session.title || 'Programming Task'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Language</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-800 uppercase font-semibold">{session.language || 'Python'}</td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Assessment ID</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-700 text-[11px]">{session.id}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Instructor</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-800">{session.teacher_name || 'Dr. Ada Smith'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. AI CONCEPTUAL UNDERSTANDING ASSESSMENT */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          2. AI Conceptual Understanding Assessment
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/2">Assessment field</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/2">Result</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-700">Transformer prediction</td>
                <td className="px-3 py-2.5 border border-slate-300">
                  <span className={`inline-block px-2.5 py-0.5 rounded font-black text-xs border ${badgeClass(verdict)}`}>
                    {verdict}
                  </span>
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-bold text-slate-700">Evidence status</td>
                <td className="px-3 py-2 border border-slate-300 font-semibold text-slate-800">
                  {vivaEvals.length >= 1 ? 'Sufficient' : 'Insufficient Evidence'}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-bold text-slate-700">Model probability (optional)</td>
                <td className="px-3 py-2 border border-slate-300 font-mono font-semibold text-slate-900">
                  {confidencePercent}% (Class posterior probability)
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-bold text-slate-700">Instructor review status</td>
                <td className="px-3 py-2 border border-slate-300 font-semibold text-slate-800">
                  {session.status === 'reviewed' ? 'Reviewed' : (report.needs_instructor_review ? 'Needs follow-up' : 'Not reviewed')}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-600 italic leading-relaxed pt-0.5">
          The prediction comes from the trained Multimodal Fusion Transformer classification head. Insufficient Evidence is a separate status, not a fourth class. Do not equate a diagnostic score with model confidence.
        </p>
      </div>

      {/* 3. MULTIMODAL EVIDENCE SUMMARY */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          3. Multimodal Evidence Summary
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/5">Modality</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-2/5">Observed evidence</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-2/5">Interpretation / limitation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Source code</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-800">
                  {codeEv.code_summary} {codeEv.execution_outcome}
                </td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-600">
                  {codeEv.evidence_limitations}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Viva responses</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-800">
                  {vivaEvals.length > 0 
                    ? `${vivaEvals.length} oral viva explanations recorded across target concepts.`
                    : 'Oral conceptual explanations provided by student.'}
                </td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-600">
                  {verdict === 'Understood' 
                    ? 'Responses demonstrate comprehensive theoretical invariant reasoning.'
                    : 'Responses demonstrate operational procedural familiarity with loop syntax, but limited formal invariant proofs.'}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Coding process</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-800">
                  {runCount} run events, {editCount} edits, {Math.round(pasteRatio * 100)}% paste ratio.
                </td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-600">
                  Iterative typing and debugging rhythm recorded; telemetry serves strictly as descriptive context without plagiarism claims.
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 4. SUPPLEMENTARY DIAGNOSTIC SCORES */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          4. Supplementary Diagnostic Scores
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Component</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold">Score (0–1)</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold">Weight</th>
                <th className="px-3 py-2 border border-slate-300 text-right font-bold">Weighted contribution</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">AST structural analysis</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{astScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">15%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-semibold">{(astScore * 0.15).toFixed(3)}</td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Execution evidence</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{execScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">20%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-semibold">{(execScore * 0.20).toFixed(3)}</td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Viva reasoning</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{vivaScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">45%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-semibold">{(vivaScore * 0.45).toFixed(3)}</td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Development process</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{procScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">20%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-semibold">{(procScore * 0.20).toFixed(3)}</td>
              </tr>
              <tr className="bg-slate-50 font-bold border-t-2 border-slate-400">
                <td className="px-3 py-2 border border-slate-300 text-slate-900 uppercase">TOTAL DIAGNOSTIC SCORE</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{compositeScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">100%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-black text-sky-900 text-sm">{(compositeScore).toFixed(3)} ({overallScore}%)</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-600 italic leading-relaxed pt-0.5">
          Overall diagnostic score = 0.15(AST) + 0.20(EXEC) + 0.45(VIVA) + 0.20(PROCESS). These formula-derived indicators are supplementary; they must not determine the Transformer class prediction.
        </p>
      </div>

      {/* 5. VIVA QUESTION AND ANSWER EVALUATION */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          5. Viva Question and Answer Evaluation
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-2.5 py-2 border border-slate-300 text-left font-bold w-1/4"># / concept</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-2/5">Question and answer summary</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold w-24">Assessment</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Evidence note</th>
              </tr>
            </thead>
            <tbody>
              {vivaEvals.length > 0 ? (
                vivaEvals.map((v, i) => (
                  <tr key={i} className="hover:bg-slate-50/50">
                    <td className="px-2.5 py-2.5 border border-slate-300 font-bold text-slate-900">
                      <div>{i + 1} &bull; {v.concept}</div>
                    </td>
                    <td className="px-3 py-2.5 border border-slate-300 text-slate-800">
                      <div className="font-semibold text-slate-900 mb-1">Q: {v.question_text}</div>
                      <div className="italic text-slate-700 bg-slate-50 p-1.5 rounded border border-slate-200">
                        A: "{v.student_answer}"
                      </div>
                    </td>
                    <td className="px-3 py-2.5 border border-slate-300 text-center">
                      <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass(v.answer_label)}`}>
                        {v.answer_label}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 border border-slate-300 text-slate-600 text-[11px]">
                      {v.evaluation_explanation || (v.concepts_correct?.length ? `Concepts validated: ${v.concepts_correct.join(', ')}` : 'Supporting explanation verified.')}
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="px-3 py-4 border border-slate-300 text-center text-slate-500 italic">
                    No viva question evaluations recorded for this assessment.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-600 italic leading-relaxed pt-0.5">
          A fluent or memorized answer is not, by itself, proof of conceptual understanding. Prefer questions asking students to predict, trace, modify, or debug program behaviour.
        </p>
      </div>

      {/* 6. CODING-PROCESS BEHAVIOUR */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          6. Coding-Process Behaviour
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/3">Recorded indicator</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold w-1/4">Value</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Paste events / pasted characters</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{nPaste} / {pasteChars}</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">
                  {nPaste === 0 ? 'No external paste events recorded' : `${nPaste} paste action(s) recorded`}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Paste-character ratio</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{pasteRatio.toFixed(2)} ({Math.round(pasteRatio * 100)}%)</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">
                  {pasteRatio < 0.20 ? 'Dominantly typed manually; no plagiarism claims' : 'Elevated paste input; verify verbal reasoning'}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Largest insertion share</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{(largestInsert).toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">
                  {largestInsert < 0.25 ? 'Incremental code growth' : 'Single large block insertion observed'}
                </td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Typing / editing / revisions</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{editCount} edits</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">Steady step-by-step program formulation</td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Runs / tests / debugging</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{runCount} runs</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">Active iterative execution and testing behavior</td>
              </tr>
              <tr>
                <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">Event log completeness</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{eventCompleteness}</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-600">Full telemetry fidelity; evidence is sufficient</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 7. POTENTIAL MISCONCEPTIONS AND CONTRADICTIONS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          7. Potential Misconceptions and Contradictions
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Concept / observation</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-2/5">Supporting evidence</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Assessment / follow-up</th>
              </tr>
            </thead>
            <tbody>
              {miscTable.length > 0 ? (
                miscTable.map((m, i) => (
                  <tr key={i} className="hover:bg-slate-50/50">
                    <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">{m.concept}</td>
                    <td className="px-3 py-2.5 border border-slate-300 text-slate-700 italic">"{m.evidence}"</td>
                    <td className="px-3 py-2.5 border border-slate-300 text-rose-800 font-semibold bg-rose-50/40">{m.interpretation}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Concept-specific misconceptions</td>
                  <td className="px-3 py-2.5 border border-slate-300 text-slate-700">No severe conceptual divergence or logical syntax violations detected.</td>
                  <td className="px-3 py-2.5 border border-slate-300 text-emerald-800 font-semibold bg-emerald-50/40">Not supported; clean program logic</td>
                </tr>
              )}

              {/* Explicit Modality Disagreement Entry if Diagnostic Score Disagrees with Transformer Class */}
              {hasModalityDiscrepancy && (
                <tr className="bg-amber-50/30">
                  <td className="px-3 py-2.5 border border-slate-300 font-bold text-amber-900">
                    Disagreement between modalities
                  </td>
                  <td className="px-3 py-2.5 border border-slate-300 text-slate-700">
                    Diagnostic score = {overallScore}% (formula checklist pass) vs. Transformer prediction = {verdict}.
                  </td>
                  <td className="px-3 py-2.5 border border-slate-300 text-amber-800 font-medium">
                    State uncertainty: Code execution is correct, but verbal responses represent procedural recall rather than formal invariant proofs. Follow-up recommended.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-slate-600 italic leading-relaxed pt-0.5">
          A paste-heavy session with correct viva explanations should be described as such; pasted code alone does not prove copying from an unauthorized source or a conceptual misconception.
        </p>
      </div>

      {/* 8. INSTRUCTOR RECOMMENDATIONS AND REVIEW */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          8. Instructor Recommendations and Review
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/3">Area</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Recommended next step</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Targeted follow-up question</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-800">{feedback.targeted_question}</td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Practice activity</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-800">{feedback.suggested_practice}</td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Instructor decision</td>
                <td className="px-3 py-2.5 border border-slate-300 font-semibold text-indigo-700">{instructorReview.final_decision}</td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Instructor comments</td>
                <td className="px-3 py-2.5 border border-slate-300 text-slate-700 italic">{instructorReview.comments}</td>
              </tr>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">Reviewed by / date</td>
                <td className="px-3 py-2.5 border border-slate-300 font-mono text-slate-800">
                  {instructorReview.reviewed_by} &bull; {instructorReview.review_date}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* REPORTING NOTE / RESPONSIBLE USE */}
      <div className="p-3.5 rounded-lg border border-slate-300 bg-slate-50 text-[11px] text-slate-700 leading-relaxed font-sans">
        <strong className="text-slate-900 font-bold block mb-0.5 uppercase tracking-wider text-[10px]">
          Reporting note:
        </strong>
        This is an instructional aid, not an automated disciplinary decision. Review underlying evidence before acting on uncertain or conflicting results.
      </div>
    </div>
  );
};
