import React, { useState, useEffect } from 'react';
import { RefreshCw, AlertCircle, ArrowRight, Sparkles } from 'lucide-react';
import { VivaInterface } from '../components/VivaInterface';
import { CodingSession, VivaQuestion } from '../types';
import { api } from '../services/api';

interface VivaViewProps {
  session: CodingSession | null;
  onNavigateToResults: () => void;
}

export const VivaView: React.FC<VivaViewProps> = ({
  session,
  onNavigateToResults,
}) => {
  const [questions, setQuestions] = useState<VivaQuestion[]>([]);
  const [activeQuestionIndex, setActiveQuestionIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const answeredCount = questions.filter((q) => q.has_answer || !!q.assessment).length;

  const fetchQuestions = async () => {
    if (!session) return;
    setIsLoading(true);
    setError(null);
    try {
      const data = await api.getVivaQuestions(session.id);
      setQuestions(data);
    } catch (err: any) {
      setError(err.message || 'Failed to load viva questions.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRegenerateQuestions = async () => {
    if (!session) return;
    setIsGenerating(true);
    setError(null);
    try {
      const newQs = await api.generateVivaQuestions(session.id);
      setQuestions(newQs);
      setActiveQuestionIndex(0);
    } catch (err: any) {
      setError(err.message || 'Failed to generate dynamic AI questions. Make sure code has been submitted.');
    } finally {
      setIsGenerating(false);
    }
  };

  useEffect(() => {
    fetchQuestions();
  }, [session]);

  const handleSubmitAnswer = async (questionId: string, answerText: string) => {
    if (!session) return;
    setIsSubmitting(true);
    setError(null);
    try {
      const assessment = await api.submitVivaAnswer(questionId, answerText);

      setQuestions((prev) =>
        prev.map((q) => {
          if (q.id === questionId) {
            return {
              ...q,
              has_answer: true,
              assessment: assessment,
            };
          }
          return q;
        })
      );
    } catch (err: any) {
      setError(err.message || 'Failed to evaluate answer.');
    } finally {
      setIsSubmitting(false);
    }
  };

  if (!session) {
    return (
      <div className="flex flex-col items-center justify-center py-20 text-center max-w-md mx-auto">
        <AlertCircle className="w-12 h-12 text-amber-500 mb-3" />
        <h3 className="text-xl font-bold text-slate-900 mb-2">No Active Session</h3>
        <p className="text-sm text-slate-500">
          Please select or start a session to access the viva interface.
        </p>
      </div>
    );
  }

  const handleViewReport = async () => {
    if (!session) return;
    try {
      await api.generateReport(session.id);
    } catch (err) {
      console.warn('Report generation notice', err);
    }
    onNavigateToResults();
  };

  return (
    <div className="max-w-7xl mx-auto space-y-6 px-4 sm:px-6 py-6">
      {/* Top Viva Header */}
      <div className="flex flex-wrap items-center justify-between gap-4 bg-white border border-slate-200 p-4 rounded-2xl shadow-sm">
        <div>
          <div className="flex items-center gap-2">
            <span className="text-xs font-mono px-2 py-0.5 rounded bg-indigo-50 text-indigo-700 border border-indigo-200 font-bold uppercase">
              Interactive Viva
            </span>
            <h2 className="text-lg font-bold text-slate-900">Dynamic AI Code Verification</h2>
          </div>
          <p className="text-xs text-slate-500 mt-0.5">
            Answer the targeted questions below to verify your conceptual comprehension of the code you submitted.
          </p>
        </div>

        <div className="flex items-center gap-3">
          <button
            onClick={handleRegenerateQuestions}
            disabled={isLoading || isGenerating}
            className="px-3.5 py-2 rounded-xl bg-gradient-to-r from-sky-50 to-indigo-50 hover:from-sky-100 hover:to-indigo-100 text-sky-800 border border-sky-300 transition-all text-xs font-bold flex items-center gap-1.5 shadow-sm"
            title="Dynamically generate brand-new AI viva questions tailored to your current code submission"
          >
            <Sparkles className={`w-3.5 h-3.5 text-sky-600 ${isGenerating ? 'animate-spin' : ''}`} />
            <span>{isGenerating ? 'Synthesizing with AI...' : 'Regenerate Questions (AI)'}</span>
          </button>

          <button
            onClick={fetchQuestions}
            disabled={isLoading || isGenerating}
            className="p-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors text-xs font-medium flex items-center gap-1.5"
            title="Refresh questions list"
          >
            <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            <span className="hidden sm:inline">Refresh</span>
          </button>

          {questions.length > 0 && answeredCount > 0 && (
            <button
              onClick={handleViewReport}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs transition-all shadow-sm"
            >
              <span>View Comprehension Report</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {error && (
        <div className="p-4 rounded-xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-center gap-2 font-medium">
          <AlertCircle className="w-4 h-4 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Progress Bar */}
      {questions.length > 0 && (
        <div className="bg-white border border-slate-200 p-3.5 rounded-xl flex items-center justify-between text-xs text-slate-700 font-mono shadow-sm">
          <div className="flex items-center gap-2 font-medium">
            <span>Viva Progress:</span>
            <strong className="text-sky-700">
              {answeredCount} / {questions.length} questions answered
            </strong>
          </div>
          <div className="w-48 h-2 bg-slate-100 rounded-full overflow-hidden border border-slate-200">
            <div
              className="h-full bg-gradient-to-r from-sky-500 to-indigo-500 transition-all duration-500"
              style={{ width: `${(answeredCount / questions.length) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Main Viva Interface Component */}
      <VivaInterface
        questions={questions}
        activeQuestionIndex={activeQuestionIndex}
        onSelectQuestion={setActiveQuestionIndex}
        onSubmitAnswer={handleSubmitAnswer}
        isSubmittingAnswer={isSubmitting}
        onRegenerateQuestions={handleRegenerateQuestions}
        isGeneratingQuestions={isGenerating}
      />
    </div>
  );
};
