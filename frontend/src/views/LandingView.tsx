import React, { useState, useEffect } from 'react';
import { 
  Cpu, 
  Code2, 
  HelpCircle, 
  ShieldCheck, 
  ArrowRight, 
  Layers, 
  TerminalSquare, 
  Activity,
  AlertTriangle,
  Sparkles
} from 'lucide-react';
import { CodingSession, AuthUser } from '../types';

interface LandingViewProps {
  onStartSession: (studentId: string, studentName: string, title: string) => Promise<void>;
  isLoading: boolean;
  onContinueCurrent: () => void;
  currentSession: CodingSession | null;
  currentUser?: AuthUser | null;
}

export const LandingView: React.FC<LandingViewProps> = ({
  onStartSession,
  isLoading,
  onContinueCurrent,
  currentSession,
  currentUser,
}) => {
  const [studentId, setStudentId] = useState(currentUser?.id || '');
  const [studentName, setStudentName] = useState(currentUser?.name || '');
  const [taskTitle, setTaskTitle] = useState('Accumulator with Modulo Filtering');

  useEffect(() => {
    if (currentUser?.id) setStudentId(currentUser.id);
    if (currentUser?.name) setStudentName(currentUser.name);
  }, [currentUser]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!studentId || !studentName) return;
    onStartSession(studentId, studentName, taskTitle);
  };

  return (
    <div className="max-w-6xl mx-auto space-y-16 py-8 px-4 sm:px-6">
      {/* Hero Banner */}
      <div className="relative text-center space-y-6 pt-6">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-sky-50 text-sky-700 border border-sky-200 text-xs font-bold uppercase tracking-wider shadow-sm">
          <Sparkles className="w-3.5 h-3.5" />
          Academic Research Platform & Verification Engine
        </div>

        <h1 className="text-4xl sm:text-6xl font-black text-slate-900 tracking-tight leading-tight">
          Verify True Comprehension.<br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-sky-600 via-indigo-600 to-cyan-600">
            Diagnose Student Misconceptions.
          </span>
        </h1>

        <p className="max-w-2xl mx-auto text-slate-600 text-base sm:text-lg leading-relaxed">
          DeepMisconception inspects Python AST structure, safely traces isolated program execution, generates dynamic viva questions grounded in student code, and cross-verifies verbal explanations to detect subtle misconceptions.
        </p>

        {/* Action Buttons */}
        <div className="flex flex-wrap items-center justify-center gap-4 pt-2">
          {currentSession && (
            <button
              onClick={onContinueCurrent}
              className="inline-flex items-center gap-2 px-6 py-3.5 rounded-xl bg-white text-sky-700 border border-sky-200 font-bold text-sm hover:bg-sky-50 transition-all shadow-sm"
            >
              <span>Resume Active Session ({currentSession.student_name})</span>
              <ArrowRight className="w-4 h-4" />
            </button>
          )}
        </div>
      </div>

      {/* Start New Session Form Card */}
      <div className="max-w-xl mx-auto bg-white border border-slate-200 rounded-3xl p-6 sm:p-8 shadow-xl relative overflow-hidden">
        <div className="absolute top-0 left-0 right-0 h-1 bg-gradient-to-r from-sky-500 to-indigo-500" />
        
        <h2 className="text-xl font-bold text-slate-900 mb-1 flex items-center gap-2">
          <Code2 className="w-5 h-5 text-sky-600" />
          Initialize Student Assessment Session
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Enter student credentials to instantiate a dedicated telemetry workspace and persistence pipeline.
        </p>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-mono font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Student ID / Roll No.
            </label>
            <input
              type="text"
              required
              value={studentId}
              onChange={(e) => setStudentId(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white"
              placeholder="e.g. STU-10492"
            />
          </div>

          <div>
            <label className="block text-xs font-mono font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Student Full Name
            </label>
            <input
              type="text"
              required
              value={studentName}
              onChange={(e) => setStudentName(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white"
              placeholder="e.g. Grace Hopper"
            />
          </div>

          <div>
            <label className="block text-xs font-mono font-semibold text-slate-700 uppercase tracking-wider mb-1">
              Problem / Task Title
            </label>
            <input
              type="text"
              required
              value={taskTitle}
              onChange={(e) => setTaskTitle(e.target.value)}
              className="w-full bg-slate-50 border border-slate-300 rounded-xl px-4 py-2.5 text-sm text-slate-900 focus:outline-none focus:ring-2 focus:ring-sky-500 focus:bg-white"
              placeholder="e.g. Recursion & Loop Invariants"
            />
          </div>

          <button
            type="submit"
            disabled={isLoading}
            className="w-full mt-2 inline-flex items-center justify-center gap-2 px-6 py-3 rounded-xl bg-gradient-to-r from-sky-600 to-indigo-600 text-white font-bold text-sm shadow-md hover:from-sky-500 hover:to-indigo-500 disabled:opacity-50 transition-all cursor-pointer"
          >
            {isLoading ? 'Creating Session...' : 'Launch Coding Workspace'}
            <ArrowRight className="w-4 h-4" />
          </button>
        </form>
      </div>

      {/* 3-Pillar Research Methodology */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 pt-4">
        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-sky-50 border border-sky-200 flex items-center justify-center text-sky-600">
            <Layers className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 text-base">1. Python AST & Safe Tracing</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Constructs concrete syntax trees to extract variables, nesting depth, and execution invariants. Executes untrusted student code inside an isolated subprocess with strict CPU, memory, and timeout bounds.
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-50 border border-indigo-200 flex items-center justify-center text-indigo-600">
            <HelpCircle className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 text-base">2. Dynamic Code-Grounded Viva</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Eliminates static quiz banks. An LLM provider formulates targeted conceptual questions directly from the student's code, probing variable state transitions, edge case handling, and architectural intent.
          </p>
        </div>

        <div className="bg-white p-6 rounded-2xl border border-slate-200 shadow-sm space-y-3">
          <div className="w-10 h-10 rounded-xl bg-emerald-50 border border-emerald-200 flex items-center justify-center text-emerald-600">
            <ShieldCheck className="w-5 h-5" />
          </div>
          <h3 className="font-bold text-slate-900 text-base">3. Evidence-Backed Diagnosis</h3>
          <p className="text-xs text-slate-600 leading-relaxed">
            Cross-verifies student verbal reasoning against concrete AST behaviors and process telemetry. Predicts <em>Understood</em>, <em>Partial Understanding</em>, or <em>Possible Misconception</em> with full transparency.
          </p>
        </div>
      </div>
    </div>
  );
};
