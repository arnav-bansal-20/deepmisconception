import React, { useState } from 'react';
import { 
  HelpCircle, 
  Send, 
  CheckCircle2, 
  XCircle, 
  AlertCircle, 
  FileCode, 
  BrainCircuit, 
  ShieldAlert
} from 'lucide-react';
import { VivaQuestion } from '../types';

interface VivaInterfaceProps {
  questions: VivaQuestion[];
  activeQuestionIndex: number;
  onSelectQuestion: (index: number) => void;
  onSubmitAnswer: (questionId: string, answerText: string) => Promise<void>;
  isSubmittingAnswer: boolean;
}

export const VivaInterface: React.FC<VivaInterfaceProps> = ({
  questions,
  activeQuestionIndex,
  onSelectQuestion,
  onSubmitAnswer,
  isSubmittingAnswer,
}) => {
  const [answerInput, setAnswerInput] = useState('');
  const activeQuestion = questions[activeQuestionIndex];

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!activeQuestion || !answerInput.trim() || isSubmittingAnswer) return;
    await onSubmitAnswer(activeQuestion.id, answerInput.trim());
    setAnswerInput('');
  };

  if (questions.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center p-12 bg-white rounded-2xl border border-slate-200 text-center max-w-xl mx-auto my-8 shadow-sm">
        <div className="w-16 h-16 rounded-2xl bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-600 mb-4 shadow-sm">
          <BrainCircuit className="w-8 h-8" />
        </div>
        <h3 className="text-xl font-bold text-slate-900 mb-2">No Viva Questions Generated Yet</h3>
        <p className="text-sm text-slate-600 mb-6 leading-relaxed">
          Submit your Python program from the Workspace to initiate the AST analysis and dynamic viva question generation pipeline.
        </p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 max-w-7xl mx-auto">
      {/* Questions Navigation Sidebar */}
      <div className="lg:col-span-4 space-y-3">
        <div className="flex items-center justify-between px-1">
          <h3 className="text-xs font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1.5">
            <HelpCircle className="w-4 h-4 text-sky-600" />
            Viva Questions ({questions.length})
          </h3>
        </div>

        <div className="space-y-2">
          {questions.map((q, idx) => {
            const isSelected = idx === activeQuestionIndex;
            const hasAnswer = q.has_answer || !!q.assessment;
            const verdict = q.assessment?.answer_label;

            return (
              <div
                key={q.id}
                onClick={() => onSelectQuestion(idx)}
                className={`p-3.5 rounded-xl cursor-pointer transition-all border text-left ${
                  isSelected
                    ? 'bg-sky-50/60 border-sky-400 shadow-sm ring-1 ring-sky-400'
                    : 'bg-white border-slate-200 hover:border-slate-300 hover:bg-slate-50/80 shadow-sm'
                }`}
              >
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-xs font-mono font-bold text-sky-700">
                    Question #{idx + 1}
                  </span>
                  <div className="flex items-center gap-1.5">
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-slate-100 text-slate-700 border border-slate-200 font-mono capitalize">
                      {q.difficulty}
                    </span>
                    {hasAnswer && (
                      <span className={`text-[10px] px-2 py-0.5 rounded-full font-bold ${
                        verdict === 'Correct'
                          ? 'bg-emerald-50 text-emerald-800 border border-emerald-300'
                          : verdict === 'Partially Correct'
                          ? 'bg-amber-50 text-amber-800 border border-amber-300'
                          : 'bg-rose-50 text-rose-800 border border-rose-300'
                      }`}>
                        {verdict || 'Answered'}
                      </span>
                    )}
                  </div>
                </div>

                <p className="text-xs text-slate-800 line-clamp-2 leading-relaxed font-medium">
                  {q.question_text}
                </p>

                <div className="mt-2 flex items-center gap-2 text-[11px] text-slate-500 font-medium">
                  <span className="text-sky-700 font-mono">Concept:</span>
                  <span className="truncate">{q.concept}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Active Question & Answer Panel */}
      <div className="lg:col-span-8 space-y-6">
        {activeQuestion && (
          <div className="bg-white rounded-2xl border border-slate-200 p-6 shadow-sm space-y-6">
            {/* Header info */}
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-slate-200 pb-4">
              <div>
                <span className="text-xs font-mono font-bold text-sky-700 uppercase tracking-wider">
                  Targeted Concept: {activeQuestion.concept}
                </span>
                <h2 className="text-lg font-bold text-slate-900 mt-1">
                  Question #{activeQuestionIndex + 1}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-xs px-2.5 py-1 rounded-full bg-slate-100 border border-slate-200 text-slate-700 font-mono">
                  Difficulty: <strong className="capitalize text-slate-900">{activeQuestion.difficulty}</strong>
                </span>
              </div>
            </div>

            {/* Question Text */}
            <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
              <p className="text-slate-800 text-base leading-relaxed font-medium">
                {activeQuestion.question_text}
              </p>
            </div>

            {/* Focal Code Snippet */}
            {activeQuestion.focal_code && (
              <div>
                <div className="flex items-center gap-1.5 text-xs text-slate-600 font-bold uppercase tracking-wider mb-2">
                  <FileCode className="w-4 h-4 text-sky-600" />
                  Referenced Code Section
                </div>
                <pre className="bg-slate-50 text-slate-900 p-4 rounded-xl font-mono text-xs border border-slate-300 overflow-x-auto">
                  {activeQuestion.focal_code}
                </pre>
              </div>
            )}

            {/* If Already Answered: Display Evaluation Assessment */}
            {activeQuestion.assessment ? (
              <div className="space-y-4 pt-4 border-t border-slate-200">
                <div className="flex items-center justify-between">
                  <h4 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
                    <CheckCircle2 className="w-4 h-4 text-emerald-600" />
                    Dynamic Answer Assessment
                  </h4>
                  <span className={`px-3 py-1 rounded-full text-xs font-bold border ${
                    activeQuestion.assessment.answer_label === 'Correct'
                      ? 'bg-emerald-50 text-emerald-800 border-emerald-300'
                      : activeQuestion.assessment.answer_label === 'Partially Correct'
                      ? 'bg-amber-50 text-amber-800 border-amber-300'
                      : 'bg-rose-50 text-rose-800 border-rose-300'
                  }`}>
                    Verdict: {activeQuestion.assessment.answer_label}
                  </span>
                </div>

                {/* Student's Answer */}
                <div className="bg-slate-50 p-4 rounded-xl border border-slate-200">
                  <span className="text-[11px] font-mono uppercase text-slate-500 font-bold block mb-1">
                    Your Response:
                  </span>
                  <p className="text-sm text-slate-800 italic leading-relaxed">
                    "{activeQuestion.assessment.student_answer}"
                  </p>
                </div>

                {/* Explanation */}
                <div className="p-4 rounded-xl bg-slate-50 border border-slate-200 text-xs text-slate-700 space-y-2">
                  <div className="font-bold text-slate-800 uppercase tracking-wider text-[11px]">
                    Evaluation Explanation:
                  </div>
                  <p className="leading-relaxed">{activeQuestion.assessment.evaluation_explanation}</p>
                </div>

                {/* Concepts Correct / Missed */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
                  {activeQuestion.assessment.concepts_correct && activeQuestion.assessment.concepts_correct.length > 0 && (
                    <div className="p-3 bg-emerald-50/60 rounded-lg border border-emerald-200">
                      <div className="font-bold text-emerald-800 flex items-center gap-1.5 mb-1.5">
                        <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600" />
                        Concepts Correctly Articulated
                      </div>
                      <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px]">
                        {activeQuestion.assessment.concepts_correct.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}

                  {activeQuestion.assessment.concepts_missed && activeQuestion.assessment.concepts_missed.length > 0 && (
                    <div className="p-3 bg-amber-50/60 rounded-lg border border-amber-200">
                      <div className="font-bold text-amber-800 flex items-center gap-1.5 mb-1.5">
                        <AlertCircle className="w-3.5 h-3.5 text-amber-600" />
                        Missing or Misunderstood Nuances
                      </div>
                      <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px]">
                        {activeQuestion.assessment.concepts_missed.map((c, i) => (
                          <li key={i}>{c}</li>
                        ))}
                      </ul>
                    </div>
                  )}
                </div>

                {activeQuestion.assessment.contradictions && activeQuestion.assessment.contradictions.length > 0 && (
                  <div className="p-3 bg-rose-50 rounded-lg border border-rose-200 text-xs">
                    <div className="font-bold text-rose-800 flex items-center gap-1.5 mb-1.5">
                      <XCircle className="w-3.5 h-3.5 text-rose-600" />
                      Direct Contradictions with Code Behavior
                    </div>
                    <ul className="list-disc list-inside space-y-1 text-slate-700 text-[11px]">
                      {activeQuestion.assessment.contradictions.map((c, i) => (
                        <li key={i}>{c}</li>
                      ))}
                    </ul>
                  </div>
                )}

                {activeQuestion.assessment.needs_instructor_review && (
                  <div className="flex items-center gap-2 p-3 bg-amber-50 rounded-lg border border-amber-200 text-xs text-amber-800">
                    <ShieldAlert className="w-4 h-4 text-amber-600 flex-shrink-0" />
                    <span>
                      Flagged for Instructor Review: The system detected ambiguity or high uncertainty in the explanation. An instructor will inspect this response.
                    </span>
                  </div>
                )}
              </div>
            ) : (
              /* Student Answer Form */
              <form onSubmit={handleSubmit} className="space-y-4 pt-4 border-t border-slate-200">
                <div>
                  <label className="block text-xs font-bold text-slate-700 uppercase tracking-wider mb-2">
                    Explain Your Reasoning In Your Own Words:
                  </label>
                  <textarea
                    rows={4}
                    value={answerInput}
                    onChange={(e) => setAnswerInput(e.target.value)}
                    placeholder="Provide a detailed explanation of what happens in this code snippet, why this construct is used, and how variables change..."
                    className="w-full bg-slate-50 border border-slate-300 rounded-xl p-3.5 text-sm text-slate-900 placeholder-slate-400 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white font-sans"
                    disabled={isSubmittingAnswer}
                  />
                </div>

                <div className="flex items-center justify-between">
                  <p className="text-xs text-slate-500 italic">
                    Answers are dynamically evaluated against the Python AST and runtime traces.
                  </p>
                  <button
                    type="submit"
                    disabled={!answerInput.trim() || isSubmittingAnswer}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 text-white font-bold text-sm shadow-md hover:from-sky-500 hover:to-indigo-500 disabled:opacity-50 disabled:cursor-not-allowed transition-all"
                  >
                    <Send className="w-4 h-4" />
                    {isSubmittingAnswer ? 'Evaluating Answer...' : 'Submit Viva Explanation'}
                  </button>
                </div>
              </form>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
