import React, { useState, useEffect, useRef } from 'react';
import { 
  Play, 
  Send, 
  Sparkles, 
  CheckCircle, 
  CheckCircle2,
  AlertCircle, 
  FileCode, 
  Activity, 
  Layers, 
  RotateCcw,
  Zap,
  BookOpen,
  Award,
  PlusCircle,
  ArrowRight,
  Lock
} from 'lucide-react';
import { MonacoEditor } from '../components/MonacoEditor';
import { ExecutionTerminal } from '../components/ExecutionTerminal';
import { ProcessTimeline } from '../components/ProcessTimeline';
import { CodingSession, CodeSubmission, ProcessEventItem, AuthUser, ProcessFeatures, Assignment } from '../types';
import { api } from '../services/api';

interface WorkspaceViewProps {
  session: CodingSession | null;
  currentUser?: AuthUser | null;
  onCodeSubmitted: (submission: CodeSubmission) => void;
  onNavigateToViva: () => void;
  onNavigateToResults?: () => void;
  onStartProblem?: (assignment: Assignment) => void;
  onSelectSession?: (session: CodingSession) => void;
}

export const WorkspaceView: React.FC<WorkspaceViewProps> = ({
  session,
  currentUser,
  onCodeSubmitted,
  onNavigateToViva,
  onNavigateToResults,
  onStartProblem,
  onSelectSession,
}) => {
  const [code, setCode] = useState('');
  const [stdout, setStdout] = useState<string | null>(null);
  const [stderr, setStderr] = useState<string | null>(null);
  const [isRunning, setIsRunning] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [executionTime, setExecutionTime] = useState<number | null>(null);
  const [telemetryQueue, setTelemetryQueue] = useState<ProcessEventItem[]>([]);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);
  const [astAnalysis, setAstAnalysis] = useState<import('../types').ASTAnalysisResult | null>(null);
  const [activeRightTab, setActiveRightTab] = useState<'terminal' | 'ast' | 'telemetry'>('terminal');
  const [processFeatures, setProcessFeatures] = useState<ProcessFeatures | null>(null);
  const [isLoadingFeatures, setIsLoadingFeatures] = useState(false);
  const [assignments, setAssignments] = useState<Assignment[]>([]);
  const [studentSessions, setStudentSessions] = useState<CodingSession[]>([]);
  const [selectedAssignment, setSelectedAssignment] = useState<Assignment | null>(null);
  const [isSelectingProblem, setIsSelectingProblem] = useState(false);

  // Load teacher-authored assignments and student history on mount
  useEffect(() => {
    const loadAssignments = async () => {
      try {
        const [list, sList] = await Promise.all([
          api.listAssignments(currentUser?.classroomId),
          currentUser?.id ? api.listSessions({ student_id: currentUser.id }) : Promise.resolve([]),
        ]);
        setAssignments(list);
        setStudentSessions(sList);
        if (list.length > 0 && !selectedAssignment) {
          // If session has title matching one of the assignments, pick it
          const matching = session ? list.find((a) => a.title.toLowerCase() === session.title.toLowerCase()) : null;
          const target = matching || list[0];
          setSelectedAssignment(target);
          setCode(target.starter_code);
        }
      } catch (err) {
        console.error('Failed to load assignments', err);
      }
    };
    loadAssignments();
  }, [currentUser?.classroomId, session?.id, currentUser?.id]);

  const getCompletedSession = (problemTitle: string) => {
    return studentSessions.find(
      (s) => s.title.toLowerCase() === problemTitle.toLowerCase() && (s.status === 'completed' || s.status === 'reviewed' || s.has_report)
    );
  };

  const handleSelectAssignment = (assignmentId: string) => {
    const found = assignments.find((a) => a.id === assignmentId);
    if (found) {
      const completed = getCompletedSession(found.title);
      if (completed) {
        alert(`Problem "${found.title}" has already been completed and evaluated. Switching to its completed report.`);
        if (onSelectSession) {
          onSelectSession(completed);
        }
        return;
      }
      setSelectedAssignment(found);
      setCode(found.starter_code);
      setStdout(null);
      setStderr(null);
      setStatusMessage(`Loaded teacher assignment: "${found.title}"`);
    }
  };

  const fetchTelemetryFeatures = async () => {
    if (!session) return;
    setIsLoadingFeatures(true);
    try {
      const feat = await api.getProcessFeatures(session.id);
      setProcessFeatures(feat);
    } catch (err) {
      console.error('Failed to load process features', err);
    } finally {
      setIsLoadingFeatures(false);
    }
  };

  // Periodically flush telemetry events
  useEffect(() => {
    if (!session || telemetryQueue.length === 0) return;

    const timer = setTimeout(async () => {
      const batch = [...telemetryQueue];
      setTelemetryQueue([]);
      try {
        await api.sendTelemetryBatch(session.id, batch);
        if (activeRightTab === 'telemetry') {
          fetchTelemetryFeatures();
        }
      } catch (err) {
        console.error('Failed to flush telemetry batch', err);
      }
    }, 2000);

    return () => clearTimeout(timer);
  }, [telemetryQueue, session, activeRightTab]);

  const handleTelemetryEvent = (event: ProcessEventItem) => {
    setTelemetryQueue((prev) => [...prev, event]);
  };

  const handleRunCode = async () => {
    if (!session) return;
    setIsRunning(true);
    setStdout(null);
    setStderr(null);
    setExecutionTime(null);
    setStatusMessage(null);

    // Record run telemetry
    handleTelemetryEvent({
      event_type: 'run',
      delta_chars: 0,
      paste_length: 0,
      timestamp: new Date().toISOString(),
    });

    try {
      const [runRes, astRes] = await Promise.all([
        api.runCode(code),
        api.analyzeCode(code),
      ]);

      setExecutionTime(runRes.execution_time_ms);
      setStdout(runRes.stdout || null);
      setStderr(runRes.stderr || null);
      setAstAnalysis(astRes);

      // Record detailed outcome event for process telemetry
      handleTelemetryEvent({
        event_type: 'run',
        delta_chars: 0,
        paste_length: 0,
        timestamp: new Date().toISOString(),
        event_metadata: {
          is_error: !runRes.success,
          message: runRes.error_type || (runRes.success ? 'Run Succeeded' : 'Run Error'),
          execution_time_ms: runRes.execution_time_ms,
        },
      });

      if (runRes.success) {
        setStatusMessage(`Executed successfully in ${runRes.execution_time_ms.toFixed(1)}ms (Isolated Sandbox)`);
      } else {
        setStatusMessage(runRes.error_type ? `Execution error: ${runRes.error_type}` : 'Execution completed with errors.');
      }
    } catch (err: any) {
      setStderr(err.message || 'Execution error encountered.');
    } finally {
      setIsRunning(false);
    }
  };

  const handleSubmitForViva = async () => {
    if (!session || isSubmitting) return;
    setIsSubmitting(true);
    setStatusMessage('Submitting code snapshot and queuing dynamic viva...');

    handleTelemetryEvent({
      event_type: 'submit',
      delta_chars: 0,
      paste_length: 0,
      timestamp: new Date().toISOString(),
    });

    try {
      if (telemetryQueue.length > 0) {
        await api.sendTelemetryBatch(session.id, telemetryQueue);
        setTelemetryQueue([]);
      }

      const submission = await api.submitCode(session.id, code);
      onCodeSubmitted(submission);
      setStatusMessage('Code submitted successfully! Proceeding to AI Viva questioning...');
      setTimeout(() => {
        onNavigateToViva();
      }, 700);
    } catch (err: any) {
      setStatusMessage(`Submission failed: ${err.message}`);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleResetCode = () => {
    if (selectedAssignment) {
      setCode(selectedAssignment.starter_code || '');
    } else {
      setCode('');
    }
    setStdout(null);
    setStderr(null);
  };

  const handleLaunchProblem = (assignment: Assignment) => {
    const completedSession = getCompletedSession(assignment.title);
    if (completedSession) {
      alert(`Problem "${assignment.title}" has already been completed and evaluated. Completed problems cannot be re-solved.`);
      return;
    }
    setSelectedAssignment(assignment);
    setCode(assignment.starter_code || '');
    setIsSelectingProblem(false);
    if (onStartProblem) {
      onStartProblem(assignment);
    }
  };

  const isCompleted = Boolean(session && (session.status === 'completed' || session.status === 'reviewed'));
  const displayAssignments = assignments;

  // If completed and not actively choosing another problem, show completed summary
  if (isCompleted && !isSelectingProblem && session) {
    return (
      <div className="max-w-2xl mx-auto py-12 px-4 space-y-6 text-center">
        <div className="bg-white border border-slate-200 p-8 rounded-2xl shadow-sm space-y-4">
          <div className="w-16 h-16 rounded-2xl bg-emerald-50 border border-emerald-200 text-emerald-600 flex items-center justify-center mx-auto shadow-sm">
            <CheckCircle2 className="w-8 h-8" />
          </div>
          <h3 className="text-xl font-black text-slate-900">
            Problem Completed: "{session.title}"
          </h3>
          <p className="text-xs text-slate-600 max-w-md mx-auto leading-relaxed">
            You have successfully completed this coding problem and finished its AI viva evaluation. Your multimodal comprehension report is ready to view.
          </p>
          <div className="pt-2 flex flex-wrap items-center justify-center gap-3">
            {onNavigateToResults && (
              <button
                onClick={onNavigateToResults}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs shadow-md transition-all"
              >
                <Award className="w-4 h-4" />
                <span>View Diagnosis Report</span>
              </button>
            )}
            <button
              onClick={() => setIsSelectingProblem(true)}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-bold text-xs shadow-md transition-all"
            >
              <PlusCircle className="w-4 h-4" />
              <span>Start Another Problem</span>
            </button>
          </div>
        </div>
      </div>
    );
  }

  // If no active session or student clicked to select/change problem, show Problem Catalog
  if (!session || isSelectingProblem) {
    return (
      <div className="max-w-6xl mx-auto py-8 px-4 sm:px-6 space-y-6">
        <div className="bg-white border border-slate-200 p-6 rounded-2xl shadow-sm flex flex-wrap items-center justify-between gap-4">
          <div>
            <div className="flex items-center gap-2">
              <span className="text-xs font-mono px-2.5 py-0.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 font-bold uppercase">
                Problem Catalog
              </span>
              {currentUser?.classroomCode && (
                <span className="text-xs text-slate-500 font-mono">
                  Group: {currentUser.classroomCode}
                </span>
              )}
            </div>
            <h2 className="text-2xl font-black text-slate-900 mt-1">
              Start a Coding Problem
            </h2>
            <p className="text-xs text-slate-500 mt-1">
              Choose an assigned problem below to start solving it in the interactive code editor.
            </p>
          </div>
          {session && !isCompleted && (
            <button
              onClick={() => setIsSelectingProblem(false)}
              className="text-xs font-bold text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 px-3.5 py-2 rounded-xl transition-colors"
            >
              Resume Current ({session.title})
            </button>
          )}
        </div>

        {displayAssignments.length > 0 ? (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
            {displayAssignments.map((a) => {
              const completedSession = getCompletedSession(a.title);
              const isCompletedProblem = Boolean(completedSession);

              return (
                <div
                  key={a.id}
                  className={`bg-white border rounded-2xl p-5 shadow-sm flex flex-col justify-between transition-all ${
                    isCompletedProblem
                      ? 'border-emerald-200 bg-emerald-50/20 shadow-none'
                      : 'border-slate-200 hover:border-sky-300 hover:shadow-md'
                  }`}
                >
                  <div className="space-y-3">
                    <div className="flex items-center justify-between gap-2">
                      <span className={`text-[11px] font-mono uppercase px-2.5 py-0.5 rounded-full font-bold border ${
                        a.difficulty === 'easy'
                          ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                          : a.difficulty === 'hard'
                          ? 'bg-rose-50 text-rose-700 border-rose-200'
                          : 'bg-amber-50 text-amber-700 border-amber-200'
                      }`}>
                        {a.difficulty}
                      </span>
                      {isCompletedProblem ? (
                        <span className="text-[10px] font-mono px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-300 font-bold flex items-center gap-1">
                          <CheckCircle2 className="w-3 h-3 text-emerald-600" />
                          Completed • {completedSession?.latest_verdict || 'Evaluated'}
                        </span>
                      ) : a.concept_tags && a.concept_tags.length > 0 ? (
                        <span className="text-[10px] font-mono text-slate-400 truncate max-w-[140px]">
                          {a.concept_tags.join(', ')}
                        </span>
                      ) : null}
                    </div>

                    <h3 className="font-bold text-slate-900 text-base leading-snug">
                      {a.title}
                    </h3>

                    <p className="text-xs text-slate-600 line-clamp-3 leading-relaxed">
                      {a.description}
                    </p>

                    {(a.sample_input || a.sample_output) && (
                      <div className="bg-slate-50 border border-slate-200 rounded-xl p-3 text-[11px] font-mono space-y-1">
                        {a.sample_input && (
                          <div className="truncate"><span className="text-slate-400 font-bold">Input:</span> {a.sample_input}</div>
                        )}
                        {a.sample_output && (
                          <div className="truncate"><span className="text-slate-400 font-bold">Output:</span> {a.sample_output}</div>
                        )}
                      </div>
                    )}
                  </div>

                  <div className="pt-4 mt-4 border-t border-slate-100">
                    {isCompletedProblem ? (
                      <div>
                        <button
                          onClick={() => {
                            if (onSelectSession && completedSession) {
                              onSelectSession(completedSession);
                            } else if (onNavigateToResults) {
                              onNavigateToResults();
                            }
                          }}
                          className="w-full inline-flex items-center justify-center gap-1.5 px-4 py-2.5 rounded-xl bg-emerald-50 hover:bg-emerald-100 text-emerald-800 font-bold text-xs border border-emerald-300 transition-all shadow-sm"
                        >
                          <Award className="w-3.5 h-3.5 text-emerald-600" />
                          <span>View Completed Report</span>
                        </button>
                        <p className="text-[10px] text-slate-500 font-medium text-center mt-1.5 flex items-center justify-center gap-1">
                          <Lock className="w-3 h-3 text-slate-400" />
                          <span>Solved & Evaluated (Locked)</span>
                        </p>
                      </div>
                    ) : (
                      <button
                        onClick={() => handleLaunchProblem(a)}
                        className="w-full inline-flex items-center justify-center gap-2 px-4 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs shadow-md shadow-sky-500/10 transition-all"
                      >
                        <Play className="w-3.5 h-3.5 fill-current" />
                        <span>Start Problem</span>
                      </button>
                    )}
                  </div>
                </div>
              );
            })}
          </div>
        ) : (
          <div className="bg-white border border-slate-200 rounded-2xl p-12 text-center">
            <p className="font-bold text-slate-800 text-sm">No assignments currently available</p>
            <p className="text-xs text-slate-500 mt-1">Assignments assigned to your classroom will appear here.</p>
          </div>
        )}
      </div>
    );
  }

  const problemName = selectedAssignment?.title || session.title || 'Coding Problem';

  const problemStatement = selectedAssignment?.description || 'Solve the programming problem according to the specifications.';

  const problemDifficulty = selectedAssignment?.difficulty || 'medium';
  const problemTags = selectedAssignment?.concept_tags && selectedAssignment.concept_tags.length > 0
    ? selectedAssignment.concept_tags
    : [];

  return (
    <div className="max-w-7xl mx-auto space-y-4 px-4 sm:px-6 py-4">
      {/* Top Workspace Header & Problem Card */}
      <div className="bg-white border border-slate-200 p-5 rounded-2xl shadow-sm space-y-3.5">
        <div className="flex flex-wrap items-center justify-between gap-4 border-b border-slate-100 pb-3">
          <div className="flex items-center gap-2 flex-wrap">
            <span className={`text-[11px] font-mono uppercase px-2.5 py-0.5 rounded-full font-bold border ${
              problemDifficulty === 'easy'
                ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                : problemDifficulty === 'hard'
                ? 'bg-rose-50 text-rose-700 border-rose-200'
                : 'bg-amber-50 text-amber-700 border-amber-200'
            }`}>
              {problemDifficulty}
            </span>
            {currentUser?.classroomCode && (
              <span className="text-[11px] font-mono px-2.5 py-0.5 rounded-full bg-indigo-50 text-indigo-800 border border-indigo-200 font-semibold">
                Group: {currentUser.classroomCode}
              </span>
            )}
            <span className="text-xs text-slate-500 font-mono">
              Candidate: <strong className="text-slate-800 font-semibold">{session.student_name}</strong>
            </span>
          </div>

          {/* Action Controls */}
          <div className="flex items-center gap-2.5 flex-wrap">
            {/* Teacher Assignment Selector */}
            {assignments.length > 0 && (
              <div className="flex items-center gap-1.5">
                <span className="text-[11px] font-mono text-slate-500 hidden sm:inline">Task:</span>
                <select
                  value={selectedAssignment?.id || ''}
                  onChange={(e) => handleSelectAssignment(e.target.value)}
                  className="bg-slate-50 border border-slate-300 rounded-xl px-3 py-2 text-xs text-slate-800 focus:outline-none focus:ring-2 focus:ring-sky-500 font-medium"
                >
                  {assignments.map((a) => {
                    const comp = getCompletedSession(a.title);
                    return (
                      <option key={a.id} value={a.id}>
                        {comp ? `✓ [Completed] ${a.title}` : `${a.title} (${a.difficulty})`}
                      </option>
                    );
                  })}
                </select>
              </div>
            )}

            <button
              onClick={() => setIsSelectingProblem(true)}
              className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors text-xs font-semibold flex items-center gap-1.5"
              title="Start another problem or view problem catalog"
            >
              <PlusCircle className="w-3.5 h-3.5 text-indigo-600" />
              <span className="hidden sm:inline">Change Problem</span>
            </button>

            <button
              onClick={handleResetCode}
              className="p-2.5 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 border border-slate-200 transition-colors text-xs font-medium flex items-center gap-1.5"
              title="Reset code template"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span className="hidden sm:inline">Reset</span>
            </button>

            <button
              onClick={handleRunCode}
              disabled={isRunning || isCompleted}
              className="inline-flex items-center gap-2 px-4 py-2.5 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs transition-colors shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Play className="w-3.5 h-3.5 fill-current" />
              {isRunning ? 'Running...' : 'Run Code'}
            </button>

            <button
              onClick={handleSubmitForViva}
              disabled={isSubmitting || isCompleted}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 hover:from-sky-500 hover:to-indigo-500 text-white font-bold text-xs transition-all shadow-sm disabled:opacity-40 disabled:cursor-not-allowed"
            >
              <Sparkles className="w-4 h-4" />
              {isSubmitting ? 'Submitting...' : isCompleted ? 'Completed (Locked)' : 'Submit for Viva'}
            </button>
          </div>
        </div>

        {/* Completed Lock Banner */}
        {isCompleted && (
          <div className="p-3 bg-emerald-50 border border-emerald-200 text-emerald-900 rounded-xl text-xs flex flex-wrap items-center justify-between gap-2 font-medium">
            <div className="flex items-center gap-2">
              <CheckCircle2 className="w-4 h-4 text-emerald-600 shrink-0" />
              <span>
                <strong>Problem Finalized:</strong> You have executed this problem and completed the AI viva evaluation. Re-solving or modifying code is locked.
              </span>
            </div>
            {onNavigateToResults && (
              <button
                onClick={onNavigateToResults}
                className="px-3 py-1 bg-white hover:bg-slate-50 text-emerald-800 font-bold text-[11px] rounded-lg border border-emerald-300 shadow-sm transition-all"
              >
                View Diagnostic Report
              </button>
            )}
          </div>
        )}

        {/* Problem Title & Statement */}
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <BookOpen className="w-4 h-4 text-sky-600 flex-shrink-0" />
            <h2 className="text-base sm:text-lg font-bold text-slate-900 tracking-tight">
              {problemName}
            </h2>
          </div>

          <div className="bg-slate-50 border border-slate-200 rounded-xl p-3.5 text-xs text-slate-700 leading-relaxed font-sans">
            <span className="font-bold text-slate-900 block mb-1 uppercase tracking-wider text-[10px] font-mono text-sky-700">
              Problem Statement:
            </span>
            <p className="text-slate-800 text-xs sm:text-sm font-medium leading-relaxed">
              {problemStatement}
            </p>
          </div>

          {(selectedAssignment?.sample_input || selectedAssignment?.sample_output) && (
            <div className="bg-sky-50/60 border border-sky-200 rounded-xl p-3 text-xs grid grid-cols-1 sm:grid-cols-2 gap-2.5 font-mono">
              {selectedAssignment.sample_input && (
                <div>
                  <span className="text-[10px] uppercase font-bold text-sky-900 block mb-1">
                    Sample Input:
                  </span>
                  <div className="bg-white px-2.5 py-1.5 rounded-lg border border-sky-200 text-slate-800 font-mono text-[11px] whitespace-pre-wrap">
                    {selectedAssignment.sample_input}
                  </div>
                </div>
              )}
              {selectedAssignment.sample_output && (
                <div>
                  <span className="text-[10px] uppercase font-bold text-emerald-800 block mb-1">
                    Expected Output:
                  </span>
                  <div className="bg-white px-2.5 py-1.5 rounded-lg border border-emerald-200 text-emerald-800 font-mono text-[11px] whitespace-pre-wrap font-semibold">
                    {selectedAssignment.sample_output}
                  </div>
                </div>
              )}
            </div>
          )}

          {problemTags.length > 0 && (
            <div className="flex flex-wrap items-center gap-1.5 pt-0.5">
              <span className="text-[10px] font-mono text-slate-500 font-bold uppercase tracking-wider mr-1">
                Target Concepts:
              </span>
              {problemTags.map((tag, i) => (
                <span
                  key={i}
                  className="text-[10px] font-mono px-2 py-0.5 rounded bg-sky-50 border border-sky-200 text-sky-700 font-medium"
                >
                  #{tag}
                </span>
              ))}
            </div>
          )}
        </div>
      </div>

      {statusMessage && (
        <div className="px-4 py-2 rounded-xl bg-sky-50 border border-sky-200 text-xs text-sky-800 flex items-center gap-2 animate-fadeIn font-medium">
          <Zap className="w-4 h-4 text-sky-600" />
          <span>{statusMessage}</span>
        </div>
      )}

      {/* Split Workspace: Editor & Output */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 h-[650px]">
        {/* Left: Monaco Code Editor */}
        <div className="lg:col-span-7 h-full flex flex-col">
          <MonacoEditor
            value={code}
            onChange={setCode}
            onTelemetryEvent={handleTelemetryEvent}
          />
        </div>

        {/* Right: Terminal & Analysis Panel */}
        <div className="lg:col-span-5 h-full flex flex-col gap-3">
          {/* Tab bar */}
          <div className="flex items-center gap-2 bg-slate-100 p-1.5 rounded-xl border border-slate-200 text-xs">
            <button
              onClick={() => setActiveRightTab('terminal')}
              className={`flex-1 py-1.5 px-2 rounded-lg font-bold transition-colors ${
                activeRightTab === 'terminal'
                  ? 'bg-white text-sky-700 border border-slate-200 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              Terminal
            </button>
            <button
              onClick={() => setActiveRightTab('ast')}
              className={`flex-1 py-1.5 px-2 rounded-lg font-bold transition-colors flex items-center justify-center gap-1.5 ${
                activeRightTab === 'ast'
                  ? 'bg-white text-indigo-700 border border-slate-200 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <span>AST</span>
              {astAnalysis && (
                <span className={`w-2 h-2 rounded-full ${astAnalysis.is_syntax_valid ? 'bg-emerald-500' : 'bg-rose-500'}`} />
              )}
            </button>
            <button
              onClick={() => {
                setActiveRightTab('telemetry');
                fetchTelemetryFeatures();
              }}
              className={`flex-1 py-1.5 px-2 rounded-lg font-bold transition-colors flex items-center justify-center gap-1.5 ${
                activeRightTab === 'telemetry'
                  ? 'bg-white text-teal-700 border border-slate-200 shadow-sm'
                  : 'text-slate-600 hover:text-slate-900'
              }`}
            >
              <Activity className="w-3.5 h-3.5 text-teal-600" />
              <span>Telemetry</span>
            </button>
          </div>

          <div className="flex-1 min-h-[350px] overflow-y-auto">
            {activeRightTab === 'terminal' ? (
              <ExecutionTerminal
                stdout={stdout}
                stderr={stderr}
                isRunning={isRunning}
                executionTimeMs={executionTime}
                sandboxUsed={true}
              />
            ) : activeRightTab === 'ast' ? (
              <div className="flex flex-col h-full bg-white rounded-xl border border-slate-300 overflow-hidden text-slate-800 shadow-sm">
                <div className="p-3 bg-slate-50 border-b border-slate-200 text-xs font-bold text-slate-800 flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Layers className="w-4 h-4 text-indigo-600" />
                    <span>Concrete Syntax Tree & Metrics</span>
                  </div>
                  {astAnalysis && (
                    <span className="font-mono text-[11px] text-slate-600">
                      Complexity: <strong className="text-slate-900">{astAnalysis.cyclomatic_complexity}</strong>
                    </span>
                  )}
                </div>

                <div className="flex-1 p-4 overflow-y-auto text-xs space-y-4 font-sans bg-white">
                  {!astAnalysis ? (
                    <div className="text-slate-400 italic py-8 text-center">
                      Run or submit your program to parse its Abstract Syntax Tree (AST).
                    </div>
                  ) : !astAnalysis.is_syntax_valid ? (
                    <div className="p-3 bg-rose-50 rounded-xl border border-rose-200 text-rose-800 space-y-1">
                      <div className="font-bold flex items-center gap-1.5">
                        <AlertCircle className="w-4 h-4 text-rose-600" />
                        <span>Syntax Error Detected (Line {astAnalysis.syntax_error_details?.line})</span>
                      </div>
                      <p className="text-[11px]">{astAnalysis.syntax_error_details?.diagnostic}</p>
                    </div>
                  ) : (
                    <div className="space-y-3">
                      {/* Functions */}
                      <div>
                        <span className="font-bold uppercase tracking-wider text-[10px] text-slate-500 block mb-1">
                          Functions ({astAnalysis.functions.length})
                        </span>
                        {astAnalysis.functions.length === 0 ? (
                          <span className="text-slate-400 italic text-[11px]">No functions declared</span>
                        ) : (
                          <div className="space-y-1">
                            {astAnalysis.functions.map((f, i) => (
                              <div key={i} className="p-2 bg-slate-50 rounded-lg border border-slate-200 font-mono text-[11px] flex items-center justify-between">
                                <span className="text-sky-700 font-bold">{f.name}({f.arguments.join(', ')})</span>
                                <span className="text-slate-500 text-[10px]">
                                  {f.is_recursive ? '🌀 Recursive' : 'Lines ' + f.line_number + '-' + f.end_line_number}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>

                      {/* Loops & Conditions */}
                      <div className="grid grid-cols-2 gap-2">
                        <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                          <span className="text-slate-500 text-[10px] block font-mono">Loops</span>
                          <span className="text-base font-bold text-slate-900 font-mono">{astAnalysis.loops.length}</span>
                        </div>
                        <div className="p-2.5 bg-slate-50 rounded-lg border border-slate-200">
                          <span className="text-slate-500 text-[10px] block font-mono">Conditions</span>
                          <span className="text-base font-bold text-slate-900 font-mono">{astAnalysis.conditions.length}</span>
                        </div>
                      </div>

                      {/* Candidate Viva Concepts */}
                      <div>
                        <span className="font-bold uppercase tracking-wider text-[10px] text-indigo-700 block mb-1.5">
                          Identified Viva Concepts ({astAnalysis.candidate_concepts.length})
                        </span>
                        <div className="space-y-1.5">
                          {astAnalysis.candidate_concepts.map((c, i) => (
                            <div key={i} className="p-2.5 bg-indigo-50/60 rounded-lg border border-indigo-200 text-[11px]">
                              <span className="font-bold text-indigo-900 block">{c.concept_name}</span>
                              <p className="text-slate-600 text-[10px] mt-0.5 line-clamp-2">{c.description}</p>
                            </div>
                          ))}
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ) : (
              <div className="h-full overflow-y-auto">
                <ProcessTimeline
                  features={processFeatures}
                  isLoading={isLoadingFeatures}
                />
              </div>
            )}
          </div>

          {/* Quick AST & Process Telemetry Info */}
          <div className="bg-white border border-slate-200 rounded-xl p-4 text-xs space-y-2 select-none shadow-sm">
            <div className="flex items-center justify-between text-slate-700 font-bold uppercase tracking-wider text-[11px]">
              <span className="flex items-center gap-1.5">
                <Layers className="w-3.5 h-3.5 text-sky-600" />
                Live Telemetry & Diagnostics
              </span>
              <span className="text-emerald-600 font-mono text-[10px] font-bold">● Recording</span>
            </div>

            <div className="grid grid-cols-3 gap-2 pt-1 font-mono text-[11px]">
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-slate-500 text-[10px]">Characters</div>
                <div className="text-slate-900 font-bold">{code.length}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-slate-500 text-[10px]">Lines</div>
                <div className="text-slate-900 font-bold">{code.split('\n').length}</div>
              </div>
              <div className="bg-slate-50 p-2 rounded-lg border border-slate-200">
                <div className="text-slate-500 text-[10px]">Queued Events</div>
                <div className="text-sky-700 font-bold">{telemetryQueue.length}</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
