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
  Compass,
  X,
  ExternalLink
} from 'lucide-react';
import { CodingSession, ComprehensionReport, ProcessFeatures, AuthUser } from '../types';
import { api } from '../services/api';
import { ProcessTimeline } from '../components/ProcessTimeline';
import { ComprehensionReportTemplate } from '../components/ComprehensionReportTemplate';

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
  const [popupSession, setPopupSession] = useState<CodingSession | null>(null);
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
    const targetId = targetSessionId || popupSession?.id || selectedSession?.id || session?.id;
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
    if (popupSession?.id) {
      loadData(false, popupSession.id);
    }
  }, [popupSession?.id]);

  const handleSelectProblem = (s: CodingSession) => {
    setSelectedSession(s);
    setPopupSession(s);
    if (onSelectSession) onSelectSession(s);
    loadData(false, s.id);
  };

  const handleClosePopup = () => {
    setPopupSession(null);
  };

  if (studentSessions.length === 0 && !session) {
    return (
      <div className="max-w-md mx-auto py-20 px-4 text-center space-y-4">
        <div className="w-12 h-12 rounded-2xl bg-sky-50 text-sky-600 border border-sky-200 flex items-center justify-center mx-auto shadow-sm">
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
      {/* Problem Selection Header & Cards */}
      <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <div>
            <h3 className="text-sm font-bold text-slate-900 uppercase tracking-wider flex items-center gap-2">
              <BookOpen className="w-4 h-4 text-sky-600" />
              <span>Assessment &amp; Misconception Reports</span>
            </h3>
            <p className="text-xs text-slate-500 mt-0.5">
              Click any problem below to inspect its comprehensive AI diagnosis report.
            </p>
          </div>
          <span className="text-xs font-mono text-slate-500 bg-slate-50 px-2.5 py-1 rounded-lg border border-slate-200 font-semibold">
            {studentSessions.length} Problem{studentSessions.length === 1 ? '' : 's'} Solved
          </span>
        </div>

        {/* Problem-wise Cards Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3 pt-2">
          {studentSessions.map((s) => {
            const isPopupOpen = popupSession?.id === s.id;
            return (
              <div
                key={s.id}
                onClick={() => handleSelectProblem(s)}
                className={`p-4 rounded-xl cursor-pointer transition-all border text-left flex flex-col justify-between hover:shadow-md ${
                  isPopupOpen
                    ? 'bg-sky-50/80 border-sky-500 ring-2 ring-sky-500/20 shadow-sm'
                    : 'bg-slate-50/70 hover:bg-white border-slate-200 hover:border-sky-300'
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
                  <h4 className="text-xs font-bold leading-snug text-slate-800 line-clamp-2">
                    {s.title}
                  </h4>
                </div>

                <div className="pt-2.5 mt-2.5 border-t border-slate-200/60 flex items-center justify-between text-[10px] text-slate-500 font-mono">
                  <span>{new Date(s.created_at).toLocaleDateString()}</span>
                  <span className="font-bold text-sky-600 flex items-center gap-1 hover:text-sky-700">
                    <span>View Report →</span>
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* POPUP MODAL VIEW FOR COMPREHENSIVE REPORT */}
      {popupSession && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50 backdrop-blur-sm animate-fadeIn overflow-y-auto">
          <div className="bg-white border border-slate-200 rounded-3xl max-w-4xl w-full p-6 sm:p-8 shadow-2xl space-y-6 my-8 max-h-[90vh] overflow-y-auto text-slate-800 relative">
            
            {/* Modal Header */}
            <div className="flex items-start justify-between border-b border-slate-200 pb-4 gap-4">
              <div>
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-xs font-mono font-bold px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 uppercase">
                    Diagnosis Report
                  </span>
                  <span className="text-xs font-mono uppercase px-2 py-0.5 rounded font-bold bg-slate-100 text-slate-700 border border-slate-200">
                    {popupSession.language}
                  </span>
                  <span className="text-xs text-slate-500 font-mono">
                    Group: <strong>{popupSession.classroom_name || 'Group Alpha'}</strong>
                  </span>
                </div>
                <h2 className="text-xl sm:text-2xl font-black text-slate-900 mt-1">
                  {popupSession.title}
                </h2>
                <p className="text-xs text-slate-500 mt-0.5">
                  Candidate: <strong className="text-slate-800">{popupSession.student_name || currentUser?.name}</strong> ({popupSession.student_id || currentUser?.id}) &bull; Solved on {new Date(popupSession.created_at).toLocaleDateString()}
                </p>
              </div>

              {/* Close Button */}
              <button
                onClick={handleClosePopup}
                className="p-2 text-slate-400 hover:text-slate-700 rounded-xl hover:bg-slate-100 transition-colors flex-shrink-0"
                title="Close popup"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            {/* Modal Body: Loading State */}
            {isLoading ? (
              <div className="py-16 text-center space-y-3">
                <RefreshCw className="w-8 h-8 text-sky-600 animate-spin mx-auto" />
                <p className="text-xs font-mono text-slate-500 font-semibold">
                  Loading diagnosis report for "{popupSession.title}"...
                </p>
              </div>
            ) : !report ? (
              /* Modal Body: Report Ready to Synthesize */
              <div className="py-8 text-center space-y-4 max-w-lg mx-auto">
                <div className="w-12 h-12 rounded-2xl bg-amber-50 border border-amber-200 text-amber-600 flex items-center justify-center mx-auto shadow-sm">
                  <Brain className="w-6 h-6" />
                </div>
                <div>
                  <h3 className="text-base font-bold text-slate-900">
                    Report Ready to Synthesize
                  </h3>
                  <p className="text-xs text-slate-500 mt-1 leading-relaxed">
                    Synthesize a multi-modal evaluation fusing your code structure, isolated execution traces, viva answer explanations, and process telemetry.
                  </p>
                </div>
                <div className="pt-2 flex items-center justify-center">
                  <button
                    onClick={() => loadData(true, popupSession.id)}
                    disabled={isRegenerating}
                    className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs transition-all shadow-md disabled:opacity-50"
                  >
                    <Sparkles className="w-4 h-4" />
                    <span>{isRegenerating ? 'Synthesizing Report...' : 'Synthesize Report Now'}</span>
                  </button>
                </div>
              </div>
            ) : (
              /* Modal Body: Complete Report Details matching 10-section template */
              <div className="space-y-6">
                <ComprehensionReportTemplate
                  report={report}
                  session={popupSession}
                  features={features}
                  currentUser={currentUser}
                  onRefresh={() => loadData(true, popupSession.id)}
                  isRegenerating={isRegenerating}
                />

                {/* Modal Footer */}
                <div className="pt-4 border-t border-slate-200 flex items-center justify-end">
                  <button
                    onClick={handleClosePopup}
                    className="px-5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs transition-colors"
                  >
                    Close
                  </button>
                </div>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
};
