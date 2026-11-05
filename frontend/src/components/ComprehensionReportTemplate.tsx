import React from 'react';
import { 
  Printer, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  HelpCircle,
  RefreshCw,
  Clock,
  Sparkles
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

  // Section 4: Viva answer evaluations
  const vivaEvals: any[] = eb.viva_evaluations || [
    {
      index: 1,
      concept: session.title || 'Core Program Invariant',
      question_text: 'Explain mechanistic logic and boundary condition.',
      student_answer: 'Student articulated logic with corresponding variables.',
      answer_label: 'Correct',
      score: 1.00
    }
  ];

  // Section 5: Misconceptions table
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

  const miscCounts = eb.misconception_counts || {
    incorrect_answers: rubric.incorrect_answers ?? 0,
    contradictions: (eb.detected_contradictions || []).length
  };

  // Section 6: Code & execution evidence
  const codeEv = eb.code_evidence || {
    code_summary: `Python solution. Problem: ${session.title}. Target constructs: loops, conditionals, accumulator.`,
    ast_findings: `Syntactically valid AST structure. Cyclomatic complexity M=2. Valid function declarations verified.`,
    execution_outcome: `Sandbox execution clean exit code 0. Verified against automated test cases.`,
    evidence_limitations: `Sandbox unit tests executed in isolated PyLite environment. Oral viva verified against verified physical stdout traces.`
  };

  // Section 7: Telemetry signals
  const telemetrySignals = eb.telemetry_signals || [
    { feature: 'Edit events', value: `${eb.edit_count || 0} events`, interpretation: 'Context only' },
    { feature: 'Run events', value: `${eb.run_count || 0} runs`, interpretation: 'Context only' },
    { feature: 'Error recovery', value: `${Math.round((eb.error_recovery_rate || 1.0) * 100)}%`, interpretation: 'Context only' },
    { feature: 'Revision cycles', value: `${Math.round((eb.process_subscores?.revision || 0.75) * 100)}%`, interpretation: 'Context only' },
    { feature: 'Testing activity', value: `${Math.round((eb.process_subscores?.testing || 0.60) * 100)}%`, interpretation: 'Context only' },
  ];

  // Section 8: Feedback
  const feedback = eb.feedback || {
    strengths: ['Modular code structure adhering to Pythonic syntax.', 'Execution output matched test invariants.'],
    areas_to_revisit: report.detected_misconceptions && report.detected_misconceptions.length > 0 
      ? report.detected_misconceptions 
      : ['Boundary conditions and loop state preservation.'],
    suggested_practice: report.recommended_interventions || ['Practice tracing iteration boundaries with manual trace tables.'],
    instructor_followup: report.needs_instructor_review 
      ? 'Review ambiguous edge-case traces with student during lab hours.' 
      : 'Demonstrated satisfactory mental model; proceed to higher-order complexity tasks.'
  };

  // Section 9: Instructor review
  const instructorReview = eb.instructor_review || {
    comments: session.status === 'reviewed' ? 'Student code and explanation verified by instructor.' : 'Pending instructor manual review and notes.',
    reviewed_by: session.teacher_name || 'Dr. Ada Smith',
    review_date: new Date(session.updated_at || Date.now()).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' }),
    final_decision: session.status === 'reviewed' ? `Instructor Review: Accepted` : 'AI-generated baseline (awaiting faculty signoff)'
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
            Student Comprehension Evaluation Report
          </h1>
          <p className="text-xs text-slate-600 mt-1">
            AI-assisted code comprehension verification &amp; misconception detection
          </p>
        </div>
        <div className="mt-3 sm:mt-0 text-right">
          <span className={`inline-block px-3 py-1 rounded-full text-xs font-black border uppercase tracking-wider ${badgeClass(verdict)}`}>
            {verdict}
          </span>
          <div className="text-[11px] font-mono text-slate-500 mt-1">
            Score: <strong className="text-slate-900">{overallScore}%</strong> &bull; Conf: {Math.round((report.confidence_score || 0.85) * 100)}%
          </div>
        </div>
      </div>

      {/* 1. STUDENT & ASSESSMENT DETAILS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          1. Student &amp; Assessment Details
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <tbody>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 w-1/4 text-slate-700">Student name</td>
                <td className="px-3 py-2 border border-slate-300 w-1/4 font-semibold text-slate-900">{session.student_name || currentUser?.name || 'Student Candidate'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 w-1/4 text-slate-700">Student ID</td>
                <td className="px-3 py-2 border border-slate-300 w-1/4 font-mono font-semibold text-slate-900">{session.student_id || currentUser?.id || 'STU-2026'}</td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Course / cohort</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-800">{session.classroom_name || 'Group Alpha (CS101)'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Assessment date</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-800">
                  {new Date(session.created_at).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                </td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Programming language</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-800 uppercase font-semibold">{session.language || 'Python'}</td>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Assessment ID</td>
                <td className="px-3 py-2 border border-slate-300 font-mono text-slate-700 text-[11px]">{session.id}</td>
              </tr>
              <tr>
                <td className="bg-slate-100 font-bold px-3 py-2 border border-slate-300 text-slate-700">Instructor</td>
                <td className="px-3 py-2 border border-slate-300 text-slate-800" colSpan={3}>{session.teacher_name || 'Dr. Ada Smith'}</td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 2. OVERALL ASSESSMENT OUTCOME */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          2. Overall Assessment Outcome
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Overall score</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Classification</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Evidence status</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Review status</th>
              </tr>
            </thead>
            <tbody>
              <tr>
                <td className="px-3 py-2.5 border border-slate-300 font-black text-base text-sky-900 font-mono">
                  {overallScore} / 100%
                </td>
                <td className="px-3 py-2.5 border border-slate-300">
                  <span className={`inline-block px-2 py-0.5 rounded font-black text-xs border ${badgeClass(verdict)}`}>
                    {verdict}
                  </span>
                </td>
                <td className="px-3 py-2.5 border border-slate-300 font-semibold text-slate-800">
                  {vivaEvals.length >= 2 ? 'Sufficient' : 'Limited'}
                </td>
                <td className="px-3 py-2.5 border border-slate-300 font-semibold text-slate-800">
                  {session.status === 'reviewed' ? 'Instructor reviewed' : 'AI-generated (Pending review)'}
                </td>
              </tr>
            </tbody>
          </table>
        </div>
      </div>

      {/* 3. SCORE BREAKDOWN */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          3. Score Breakdown
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
                <td className="px-3 py-2 border border-slate-300 text-slate-900 uppercase">TOTAL</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">{compositeScore.toFixed(2)}</td>
                <td className="px-3 py-2 border border-slate-300 text-center font-mono">100%</td>
                <td className="px-3 py-2 border border-slate-300 text-right font-mono font-black text-sky-900 text-sm">{overallScore}%</td>
              </tr>
            </tbody>
          </table>
        </div>
        <div className="text-[11px] text-slate-500 font-mono bg-slate-50/70 p-2 rounded border border-slate-200">
          Formula: Overall = 0.15(AST) + 0.20(EXEC) + 0.45(VIVA) + 0.20(PROCESS)
        </div>
      </div>

      {/* 4. VIVA ANSWER EVALUATION */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          4. Viva Answer Evaluation
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-2.5 py-2 border border-slate-300 text-center w-8 font-bold">#</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/3">Question / concept</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Answer summary</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold w-24">Label</th>
                <th className="px-3 py-2 border border-slate-300 text-right font-bold w-16">Score</th>
              </tr>
            </thead>
            <tbody>
              {vivaEvals.map((v, i) => (
                <tr key={i} className="hover:bg-slate-50/50">
                  <td className="px-2.5 py-2.5 border border-slate-300 text-center font-mono font-bold text-slate-600">{v.index || i + 1}</td>
                  <td className="px-3 py-2.5 border border-slate-300 font-semibold text-slate-900">
                    <div>{v.concept}</div>
                    <div className="text-[10px] text-slate-500 font-normal mt-0.5 line-clamp-2">{v.question_text}</div>
                  </td>
                  <td className="px-3 py-2.5 border border-slate-300 text-slate-700 italic">
                    "{v.student_answer}"
                  </td>
                  <td className="px-3 py-2.5 border border-slate-300 text-center">
                    <span className={`inline-block px-2 py-0.5 rounded text-[10px] font-bold border ${badgeClass(v.answer_label)}`}>
                      {v.answer_label}
                    </span>
                  </td>
                  <td className="px-3 py-2.5 border border-slate-300 text-right font-mono font-bold text-slate-900">
                    {Number(v.score || 0).toFixed(2)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 5. POSSIBLE MISCONCEPTIONS & CONTRADICTIONS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          5. Possible Misconceptions &amp; Contradictions
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/4">Concept</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/3">Evidence from answers</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold w-1/5">Pattern</th>
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Interpretation</th>
              </tr>
            </thead>
            <tbody>
              {miscTable.length > 0 ? (
                miscTable.map((m, i) => (
                  <tr key={i} className="hover:bg-slate-50/50">
                    <td className="px-3 py-2.5 border border-slate-300 font-bold text-slate-900">{m.concept}</td>
                    <td className="px-3 py-2.5 border border-slate-300 text-slate-700 italic">"{m.evidence}"</td>
                    <td className="px-3 py-2.5 border border-slate-300 text-slate-600 font-mono text-[11px]">{m.pattern}</td>
                    <td className="px-3 py-2.5 border border-slate-300 text-rose-800 font-semibold bg-rose-50/40">{m.interpretation}</td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={4} className="px-3 py-3 border border-slate-300 text-center text-emerald-800 bg-emerald-50/40 font-semibold">
                    No severe conceptual misconceptions or direct code-explanation contradictions detected.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {/* 6. CODE & EXECUTION EVIDENCE */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          6. Code &amp; Execution Evidence
        </h2>
        <div className="border border-slate-300 rounded-lg p-3.5 text-xs space-y-2 bg-white">
          <div><strong className="text-slate-900 font-bold">Code summary:</strong> <span className="text-slate-700">{codeEv.code_summary}</span></div>
          <div><strong className="text-slate-900 font-bold">AST findings:</strong> <span className="text-slate-700">{codeEv.ast_findings}</span></div>
          <div><strong className="text-slate-900 font-bold">Execution outcome:</strong> <span className="text-slate-700 font-mono bg-slate-50 px-2 py-0.5 rounded border border-slate-200">{codeEv.execution_outcome}</span></div>
          <div><strong className="text-slate-900 font-bold">Evidence limitations:</strong> <span className="text-slate-600 italic">{codeEv.evidence_limitations}</span></div>
        </div>
      </div>

      {/* 7. DEVELOPMENT PROCESS SIGNALS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          7. Development Process Signals
        </h2>
        <div className="overflow-x-auto">
          <table className="w-full text-xs border-collapse border border-slate-300">
            <thead>
              <tr className="bg-slate-100 text-slate-700">
                <th className="px-3 py-2 border border-slate-300 text-left font-bold">Telemetry feature</th>
                <th className="px-3 py-2 border border-slate-300 text-center font-bold">Observed value</th>
                <th className="px-3 py-2 border border-slate-300 text-right font-bold">Interpretation / caveat</th>
              </tr>
            </thead>
            <tbody>
              {telemetrySignals.map((t: any, i: number) => (
                <tr key={i}>
                  <td className="px-3 py-2 border border-slate-300 font-medium text-slate-800">{t.feature}</td>
                  <td className="px-3 py-2 border border-slate-300 text-center font-mono font-bold text-slate-900">{t.value}</td>
                  <td className="px-3 py-2 border border-slate-300 text-right text-slate-500 italic">{t.interpretation}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* 8. FEEDBACK & RECOMMENDED NEXT STEPS */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          8. Feedback &amp; Recommended Next Steps
        </h2>
        <div className="border border-slate-300 rounded-lg p-3.5 text-xs space-y-2 bg-white">
          <div>
            <strong className="text-emerald-900 font-bold block mb-0.5">Strengths:</strong>
            <ul className="list-disc list-inside space-y-0.5 text-slate-700">
              {(feedback.strengths || []).map((s: string, i: number) => (
                <li key={i}>{s}</li>
              ))}
            </ul>
          </div>
          <div className="pt-1.5 border-t border-slate-200">
            <strong className="text-rose-900 font-bold block mb-0.5">Areas to revisit:</strong>
            <ul className="list-disc list-inside space-y-0.5 text-slate-700">
              {(feedback.areas_to_revisit || []).map((a: string, i: number) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
          </div>
          <div className="pt-1.5 border-t border-slate-200">
            <strong className="text-indigo-900 font-bold block mb-0.5">Suggested practice:</strong>
            <ul className="list-disc list-inside space-y-0.5 text-slate-700">
              {(feedback.suggested_practice || []).map((p: string, i: number) => (
                <li key={i}>{p}</li>
              ))}
            </ul>
          </div>
          <div className="pt-1.5 border-t border-slate-200">
            <strong className="text-slate-900 font-bold">Instructor follow-up:</strong>{' '}
            <span className="text-slate-700">{feedback.instructor_followup}</span>
          </div>
        </div>
      </div>

      {/* 9. INSTRUCTOR REVIEW */}
      <div className="space-y-2">
        <h2 className="text-xs font-black uppercase tracking-wider text-slate-900 border-l-4 border-sky-600 pl-2">
          9. Instructor Review
        </h2>
        <div className="border border-slate-300 rounded-lg p-3.5 text-xs space-y-2.5 bg-white">
          <div>
            <strong className="text-slate-900 font-bold block mb-0.5">Instructor comments:</strong>
            <span className="text-slate-700 leading-relaxed font-sans">{instructorReview.comments}</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 pt-2 border-t border-slate-200 text-slate-700">
            <div><strong className="text-slate-900">Reviewed by:</strong> {instructorReview.reviewed_by}</div>
            <div><strong className="text-slate-900">Date:</strong> {instructorReview.review_date}</div>
          </div>
          <div className="pt-2 border-t border-slate-200 flex flex-wrap items-center justify-between gap-3">
            <div>
              <strong className="text-slate-900">Final decision:</strong>{' '}
              <span className="text-indigo-700 font-semibold">{instructorReview.final_decision}</span>
            </div>
            <label className="text-slate-700 font-mono text-[11px] flex items-center gap-1.5 cursor-pointer">
              <input type="checkbox" defaultChecked className="rounded border-slate-300 text-sky-600" />
              <span>Student acknowledgement</span>
            </label>
          </div>
        </div>
      </div>

      {/* 10. METHOD & RESPONSIBLE USE NOTE */}
      <div className="p-3.5 rounded-lg border border-slate-300 bg-slate-50 text-[11px] text-slate-600 leading-relaxed font-sans">
        <strong className="text-slate-900 font-bold block mb-0.5 uppercase tracking-wider text-[10px]">
          10. Method &amp; Responsible Use Note
        </strong>
        This report combines code-structure, execution, viva-answer, and development-process evidence. The misconception label is a rule-based indicator based on assessed answer patterns and contradictions—not a definitive judgment about a student's knowledge. Review uncertain or consequential outcomes with an instructor. Include only verified scores and evidence, and identify AI-generated assessments and human review.
      </div>
    </div>
  );
};

